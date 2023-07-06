import os.path
import string, time
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import logging
import copy
import torch
import statistics, math
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import PIL
import sys
sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/DAVE2-Keras')
from DAVE2pytorch import DAVE2PytorchModel, DAVE2v3
# import VAEsteer, VAE, VAEbasic
# from VAEsteer import *
# sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/superdeepbillboard')
sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/BeamNGpy')
sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/BeamNGpy/src/')
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
from beamngpy import ProceduralCube
# sys.path.append(f'{args.path2src}/GitHub/superdeepbillboard')
# sys.path.append(f'{args.path2src}/GitHub/BeamNGpy')
# sys.path.append(f'{args.path2src}/GitHub/BeamNGpy/src/')
# from wand.image import Image as WandImage
from sim_utils import *

# globals
integral, prev_error = 0.0, 0.0
overall_throttle_setpoint = 40
setpoint = overall_throttle_setpoint
lanewidth = 3.75 #2.25
centerline, centerline_interpolated = [], []
roadleft, roadright = [], []
episode_steps, interventions = 0, 0
training_file = ""
topo_id = None
steer_integral, steer_prev_error = 0., 0.
scenario_name = ""
parentdir = "F:/supervised-transformation-dataset-alltransforms3"

if not os.path.exists(parentdir):
    os.makedirs(parentdir, exist_ok=True)

def setup_sensors(vehicle, img_dims, fov=51):
    # fov = fov # 60 works for full lap #63 breaks on hairpin turn
    resolution = img_dims
    pos = (-0.5, 0.38, 1.3)
    direction = (0, 1.0, 0)
    front_camera = Camera(pos, direction, fov, resolution,
                          colour=True, depth=True, annotation=True)
    base_camera = Camera(pos, direction, 51, (240, 135),
                          colour=True, depth=True, annotation=True)
    hires_camera = Camera(pos, direction, 51, (480, 270),
                          colour=True, depth=True, annotation=True)
    lores_camera = Camera(pos, direction, 51, (96, 54),
                          colour=True, depth=True, annotation=True)

    gforces = GForces()
    electrics = Electrics()
    damage = Damage()
    timer = Timer()
    vehicle.attach_sensor("base_cam", base_camera)
    vehicle.attach_sensor('front_cam', front_camera)
    vehicle.attach_sensor('hires_cam', hires_camera)
    vehicle.attach_sensor('lores_cam', lores_camera)
    vehicle.attach_sensor('gforces', gforces)
    vehicle.attach_sensor('electrics', electrics)
    vehicle.attach_sensor('damage', damage)
    vehicle.attach_sensor('timer', timer)
    return vehicle


def throttle_PID(kph, dt):
    global integral, prev_error, setpoint
    kp = 0.19; ki = 0.0001; kd = 0.008
    error = setpoint - kph
    if dt > 0:
        deriv = (error - prev_error) / dt
    else:
        deriv = 0
    integral = integral + error * dt
    w = kp * error + ki * integral + kd * deriv
    prev_error = error
    return w


''' takes in 3D array of sequential [x,y] '''
def plot_deviation(trajectories, model, deflation_pattern, savefile="trajectories"):
    global centerline, roadleft, roadright
    x, y = [], []
    for point in centerline:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k-")
    x, y = [], []
    for point in roadleft:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k-")
    x, y = [], []
    for point in roadright:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k-", label="Road")
    for i,t in enumerate(trajectories):
        x,y = [],[]
        for point in t:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, label="Run {}".format(i))
    # if "winding" in savefile:
    #     plt.xlim([700, 900])
    #     plt.ylim([-150, 50])
    # elif "straight" in savefile:
    #     plt.xlim([50, 180])
    #     plt.ylim([-300, -260])
    # elif "Rturn" in savefile:
    #     plt.xlim([250, 400])
    #     plt.ylim([-300, -150])
    # elif "Lturn" in savefile:
    #     plt.xlim([-400, -250])
    #     plt.ylim([-850, -700])
    plt.title(f'Trajectories with {model} \n{savefile}')
    plt.legend()
    plt.draw()
    plt.savefig(f"{deflation_pattern}/{savefile}.jpg")
    plt.show()
    plt.pause(0.1)


def calc_deviation_from_center(centerline, traj):
    dists = []
    for point in traj:
        dist = dist_from_line(centerline, point)
        dists.append(min(dist))
    stddev = statistics.stdev(dists)
    avg = sum(dists) / len(dists)
    return {"stddev":stddev, "mean":avg}

def plot_racetrack_roads(bng, road_id, seg=None, reverse=False):
    global scenario_name
    print("Plotting scenario roads...")
    roads = bng.get_roads()
    print(f"retrieved roads ({len(roads.keys())} total)")
    sp = spawn_point(scenario_name, road_id, seg=seg, reverse=reverse)
    colors = ['b','g','r','c','m','y','k']
    symbs = ['-','--','-.',':','.',',','v','o','1',]
    selectedroads = []
    print("iterating over roads")
    for road in roads:
        print(road, roads[road]['drivability'])
        if float(roads[road]['drivability']) >= 1:
            road_edges = bng.get_road_edges(road)
            xy_def = np.array([edge['middle'][:2] for edge in road_edges])
            xydef1 = xy_def[:-1] -xy_def[1:]
            xydef1 = np.square(xydef1)
            xydef1 = np.sum(xydef1, axis=1)
            xydef1 = np.sqrt(xydef1)
            s = np.sum(xydef1)
            # xy_def = [edge['middle'][:2] for edge in road_edges]
            # dists = [distance(xy_def[i], xy_def[i+1]) for i,p in enumerate(xy_def[:-1])]
            # s = sum(dists)
            # if (s < 200 or s > 300):
            if (s < 75):
                continue
            x_temp, y_temp = [], []
            for edge in road_edges:
                x_temp.append(edge['middle'][0])
                y_temp.append(edge['middle'][1])
                symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
            plt.plot(x_temp, y_temp, symb, label=road)
            print(road, road_edges[0]['middle'], roads[road]['drivability'], s)
            # selectedroads.append(road)
    # for r in selectedroads: # ["8179", "8248", "8357", "8185", "7770", "7905", "8205", "8353", "8287", "7800", "8341", "7998"]:
    #     a = bng.get_road_edges(r)
    #     print(r, a[0]['middle'], roads[r]['drivability'])
    plt.plot([sp['pos'][0]], [sp['pos'][1]], "bo")
    plt.legend(ncol=10)
    plt.show()
    plt.pause(0.001)

def get_nearby_racetrack_roads(bng, point_of_in, default_scenario):
    print(f"Plotting nearby roads to point={point_of_in}")
    roads = bng.get_roads()
    print("retrieved roads")
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    symbs = ['-', '--', '-.', ':', '.', ',', 'v', 'o', '1', ]
    for road in roads:
        road_edges = bng.get_road_edges(road)
        x_temp, y_temp = [], []
        if len(road_edges) < 20:
            continue
        xy_def = [edge['middle'][:2] for edge in road_edges]
        # dists = [distance(xy_def[i], xy_def[i + 1]) for i, p in enumerate(xy_def[:-1:5])]
        # road_len = sum(dists)
        # find minimum distance from point of interest
        dists = [distance(i, point_of_in) for i in xy_def]
        s = min(dists)
        if (s > 200): # or road_len < 200:
            continue
        for edge in road_edges:
            x_temp.append(edge['middle'][0])
            y_temp.append(edge['middle'][1])
        symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
        plt.plot(x_temp, y_temp, symb, label=road)
        print(f"{road=}\tstart=({x_temp[0]},{y_temp[0]},{road_edges[0]['middle'][2]})\t{road_edges[0]['middle']}")
    plt.plot([point_of_in[0]], [point_of_in[1]], "bo")
    plt.title(f"{default_scenario} poi={point_of_in}")
    plt.legend(ncol=10)
    plt.draw()
    plt.savefig(f"points near {point_of_in}.jpg")
    plt.show()
    plt.pause(0.001)


def road_analysis(bng, road_id, seg=None, reverse=False):
    global centerline, roadleft, roadright, scenario_name
    print("Performing road analysis...")
    # plot_racetrack_roads(bng, road_id, seg=seg)
    # get_nearby_racetrack_roads(bng, spawn_point(scenario_name, road_id)['pos'], scenario_name)
    print(f"Getting road {road_id}...")
    edges = bng.get_road_edges(road_id)
    if reverse:
        edges.reverse()
        print(f"new spawn={edges[0]['middle']}")
    else:
        print(f"reversed spawn={edges[-1]['middle']}")
    centerline = [edge['middle'] for edge in edges]
    print(centerline[0:10])
    if road_id == "8185":
        edges = bng.get_road_edges("8096")
        roadleft = [edge['middle'] for edge in edges]
        edges = bng.get_road_edges("7878") # 7820, 7878, 7805
        roadright = [edge['middle'] for edge in edges]
    else:
        roadleft = [edge['left'] for edge in edges]
        roadright = [edge['right'] for edge in edges]

    # with open(f"road-def-{scenario_name}-{road_id}.txt", "w") as f:
    #     f.write("CENTER\n")
    #     for p in centerline:
    #         p = str(p).replace(", ", ",")
    #         p = p.replace("]","").replace("[","")
    #         f.write(f"{p}\n")
    #     f.write("LEFT\n")
    #     for p in roadleft:
    #         p = str(p).replace(", ", ",")
    #         p = p.replace("]", "").replace("[", "")
    #         f.write(f"{p}\n")
    #     f.write("RIGHT\n")
    #     for p in roadright:
    #         p = str(p).replace(", ", ",")
    #         p = p.replace("]", "").replace("[", "")
    #         f.write(f"{p}\n")
    # exit(0)
    return centerline

def plot_trajectory(traj, title="Trajectory", label1="car traj."):
    global centerline, roadleft, roadright
    plt.plot([t[0] for t in centerline], [t[1] for t in centerline], 'r-')
    plt.plot([t[0] for t in roadleft], [t[1] for t in roadleft], 'r-')
    plt.plot([t[0] for t in roadright], [t[1] for t in roadright], 'r-')
    x = [t[0] for t in traj]
    y = [t[1] for t in traj]
    plt.plot(x,y, 'b--', label=label1)
    plt.title(title)
    plt.legend()
    plt.draw()
    plt.savefig(f"{title}.jpg")
    plt.show()
    plt.pause(0.1)

def plot_input(timestamps, input, input_type, run_number=0):
    plt.plot(timestamps, input)
    plt.xlabel('Timestamps')
    plt.ylabel('{} input'.format(input_type))
    plt.title("{} over time".format(input_type))
    plt.savefig("Run-{}-{}.png".format(run_number, input_type))
    plt.show()
    plt.pause(0.1)

def create_ai_line_from_road_with_interpolation(spawn, bng, road_id, seg=None, reverse=False):
    global centerline, roadleft, roadright, centerline_interpolated
    points, point_colors, spheres, sphere_colors = [], [], [], []
    centerline_interpolated = []
    road_analysis(bng, road_id, seg=seg, reverse=reverse)
    # interpolate centerline at 1m distance
    for i, p in enumerate(centerline[:-1]):
        if distance(p, centerline[i + 1]) > 1:
            y_interp = interpolate.interp1d([p[0], centerline[i + 1][0]], [p[1], centerline[i + 1][1]])
            num = int(distance(p, centerline[i + 1]))
            xs = np.linspace(p[0], centerline[i + 1][0], num=num, endpoint=True)
            ys = y_interp(xs)
            for x, y in zip(xs, ys):
                centerline_interpolated.append([x, y])
        else:
            centerline_interpolated.append([p[0], p[1]])
    # set up debug line
    for p in centerline[:-1]:
        points.append([p[0], p[1], p[2]])
        point_colors.append([0, 1, 0, 0.1])
        spheres.append([p[0], p[1], p[2], 0.25])
        sphere_colors.append([1, 0, 0, 0.8])
    bng.add_debug_line(points, point_colors, spheres=spheres, sphere_colors=sphere_colors, cling=True, offset=0.1)


def add_barriers(scenario):
    with open('posefiles/industrial_racetrack_barrier_locations.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            pos = line[0].split(',')
            pos = tuple([float(i) for i in pos])
            rot_quat = line[1].split(',')
            rot_quat = tuple([float(j) for j in rot_quat])
            # turn barrier 90 degrees
            r = R.from_quat(list(rot_quat))
            r = r.as_euler('xyz', degrees=True)
            r[2] = r[2] + 90
            r = R.from_euler('xyz', r, degrees=True)
            rot_quat = tuple(r.as_quat())
            barrier = StaticObject(name='barrier{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(1, 1, 1),
                                shape='levels/Industrial/art/shapes/misc/concrete_road_barrier_a.dae')
            # barrier.type="BeamNGVehicle"
            scenario.add_object(barrier)


def setup_beamng(default_scenario, road_id, transf="None", reverse=False, seg=None, img_dims=(240,135), fov=51, vehicle_model='etk800', default_color="green", steps_per_sec=15,
                 beamnginstance='C:/Users/Meriel/Documents/BeamNG.researchINSTANCE4', port=64956):
    global scenario_name
    scenario_name = default_scenario
    random.seed(1703)
    setup_logging()
    beamng = BeamNGpy('localhost', port, home='C:/Users/Meriel/Documents/BeamNG.research.v1.7.0.1', user=beamnginstance)
    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model, licence='EGO', color=default_color)
    vehicle = setup_sensors(vehicle, img_dims, fov=fov)
    print(default_scenario, road_id, reverse, seg)
    spawn = spawn_point(default_scenario, road_id, reverse=reverse, seg=seg)
    print(spawn)
    scenario.add_vehicle(vehicle, pos=spawn['pos'], rot=None, rot_quat=spawn['rot_quat']) #, partConfig=parts_config)
    if default_scenario == "industrial":
        add_barriers(scenario)
    scenario.make(beamng)
    bng = beamng.open(launch=True)
    bng.set_deterministic()
    bng.set_steps_per_second(steps_per_sec)
    bng.load_scenario(scenario)
    bng.start_scenario()
    create_ai_line_from_road_with_interpolation(spawn, bng, road_id, seg=seg, reverse=reverse)
    bng.pause()
    assert vehicle.skt
    return vehicle, bng, scenario

def run_scenario(vehicle, bng, scenario, model, default_scenario, road_id, transf="None", reverse=False, vehicle_model='etk800', run_number=0,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), seg=None, hash="hash"):
    global integral, prev_error, setpoint, steer_prev_error
    global episode_steps, interventions
    global parentdir
    hash = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    cutoff_point = spawn_point(default_scenario, road_id, reverse=reverse, seg=seg)['pos']
    bng.restart_scenario()
    vehicle.update_vehicle()
    wheelspeed, kph, throttle, integral, runtime, damage = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    kphs, traj, steering_inputs, throttle_inputs, timestamps = [], [], [], [], []
    frames_adjusted, episode_steps = 0, 0
    prev_error = setpoint
    reached_cutoff = False
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    writedir = f"{parentdir}/{default_scenario}-{road_id}-{topo_id}-{transf}.{seg}-run{run_number:02d}-{timestr}-{hash}"
    if not os.path.isdir(writedir):
        # os.mkdir(writedir)
        os.makedirs(writedir, exist_ok=True)
    with open(f"{writedir}/data.txt", "w") as f:
        f.write(f"IMG,PREDICTION,STEERING_INPUT,POSITION,ORIENTATION,KPH,STEERING_ANGLE_CURRENT,THROTTLE_INPUT\n")
        while kph < 38:
            vehicle.update_vehicle()
            sensors = bng.poll_sensors(vehicle)
            last_time = sensors['timer']['time']
            start_time = sensors['timer']['time']
            kph = ms_to_kph(sensors['electrics']['wheelspeed'])
            vehicle.control(throttle=1., steering=0., brake=0.0)
            bng.step(1, wait=True)
            outside_track, distance_from_center, leftrightcenter, segment_shape, theta_deg = has_car_left_track(vehicle)
        while damage <= 1:
            vehicle.update_vehicle()
            sensors = bng.poll_sensors(vehicle)
            image = sensors['front_cam']['colour'].convert('RGB')
            image_base = sensors['base_cam']['colour'].convert('RGB')
            image_depth = sensors['base_cam']['depth'].convert('RGB')
            image_lores = sensors['lores_cam']['colour'].convert('RGB')
            image_hires = sensors['hires_cam']['colour'].convert('RGB')

            # if "fisheye" in transf:
            #     image = fisheye_inv(image)
            # elif "resdec" in transf or "resinc" in transf:
            #     image = image.resize((240,135))
            #     # image = cv2.resize(np.array(image), (135,240))
            # elif "depth" in transf:
            #     image_seg = sensors['front_cam']['annotation'].convert('RGB')

            cv2.imshow('car view', np.array(image)[:, :, ::-1])
            cv2.imshow('depthimg', np.array(image_depth))
            cv2.waitKey(1)
            kph = ms_to_kph(sensors['electrics']['wheelspeed'])
            dt = sensors['timer']['time'] - last_time
            episode_steps += 1
            runtime = sensors['timer']['time'] - start_time

            processed_img = model.process_image(image).to(device)
            base_model_inf = model(processed_img)
            base_model_inf = float(base_model_inf.item())
            curr_steering = sensors['electrics']['steering_input']
            # expert_action, cartocl_theta_deg = get_expert_action(vehicle)
            expert_action = -leftrightcenter * (distance_from_center / 8)
            if "Rturn" in topo_id or "Lturn" in topo_id or "extra" in topo_id:
                expert_action = -leftrightcenter * (distance_from_center)
            # print(f"action={expert_action=:.3f}\t\ttheta{math.degrees(cartocl_theta_deg)=:.3f}")
            # evaluation = abs(expert_action - base_model_inf) < 0.05
            # if evaluation:
            #     steering = base_model_inf
            #     blackedout = np.ones((100,100,3))
            #     blackedout[:, :, :2] = blackedout[:, :, :2] * 0
            #     cv2.imshow("action image", blackedout)  # red
            #     cv2.waitKey(1)
            # else:
            setpoint_steering = expert_action
            steering = steering_PID(curr_steering, setpoint_steering, dt)
            cv2.imshow("action image", np.zeros((120,120,3)))  # black
            cv2.waitKey(1)
            frames_adjusted += 1

            if abs(steering) > 0.15:
                setpoint = 30
            else:
                setpoint = 40

            throttle = throttle_PID(kph, dt)
            vehicle.control(throttle=throttle, steering=steering, brake=0.0)
            steering_inputs.append(steering)
            throttle_inputs.append(throttle)
            timestamps.append(runtime)
            damage = sensors['damage']["damage"]

            kph = ms_to_kph(sensors['electrics']['wheelspeed'])
            dt = (sensors['timer']['time'] - start_time) - runtime
            position = str(vehicle.state['pos']).replace(",", " ")
            orientation = str(vehicle.state['dir']).replace(",", " ")
            runtime = sensors['timer']['time'] - start_time
            image.save(f"{writedir}/sample-transf-{episode_steps:05d}.jpg", "JPEG")
            image_base.save(f"{writedir}/sample-base-{episode_steps:05d}.jpg", "JPEG")
            image_hires.save(f"{writedir}/sample-hires-{episode_steps:05d}.jpg", "JPEG")
            image_lores.save(f"{writedir}/sample-lores-{episode_steps:05d}.jpg", "JPEG")
            image_depth.save(f"{writedir}/sample-depth-{episode_steps:05d}.jpg", "JPEG")


            vehicle.update_vehicle()
            traj.append(vehicle.state['pos'])

            kphs.append(ms_to_kph(wheelspeed))
            # dists = dist_from_line(centerline, vehicle.state['pos'])

            if damage > 1.0:
                print(f"Damage={damage:.3f}, exiting...")
                break
            last_time = sensors['timer']['time']
            bng.step(1, wait=False)
            vehicle.update_vehicle()
            sensors = bng.poll_sensors(vehicle)
            f.write(f"sample-base-{episode_steps:05d}.jpg,{steering},{sensors['electrics']['steering_input']},{position},{orientation},{kph},{sensors['electrics']['steering']},{throttle}\n")
            # if distance2D(vehicle.state["pos"], cutoff_point) < 12 and sensors['timer']['time'] > 50:
            #     reached_cutoff = True
            #     print("Reached cutoff point, exiting...")
            #     break

            outside_track, distance_from_center, leftrightcenter, segment_shape, theta_deg = has_car_left_track(vehicle)
            if outside_track:
                print("Left track, exiting...")
                break

    cv2.destroyAllWindows()

    deviation = calc_deviation_from_center(centerline, traj)
    results = {'runtime': round(runtime,3), 'damage': damage, 'kphs':kphs, 'traj':traj,
               'deviation':deviation, "interventions":frames_adjusted, "episode_steps":episode_steps,
               "reached_cutoff": reached_cutoff, "outside_track": outside_track, "inputs": steering_inputs
               }
    return results

def steering_PID(curr_steering,  steer_setpoint, dt):
    global steer_integral, steer_prev_error, topo_id
    if dt == 0:
        return 0
    if "winding" in topo_id:
        # kp = .4; ki = 0.00; kd = 0.03
        #kp = 3; ki = 0.00; kd = 0.01 # using angle relative to ctrline
        #kp = 0.75; ki = 0.00; kd = 0.2 # using LRC and dist to ctrline
        #kp = 0.5; ki = 0.00; kd = 0.2  # using LRC and dist to ctrline; avg dist from center=1.0809121348292758
        #OLDTUNING kp = 0.425; ki = 0.00; kd = 0.2  # using LRC and dist to ctrline; Average deviation: 1.023
        kp = 1.2; ki = 0.00; kd = 0.4
        # kp = 0.225; ki = 0.00; kd = 0.1  # using LRC and dist to ctrline; Average deviation:
    elif "straight" in topo_id:
        # kp = 0.8125; ki = 0.00; kd = 0.2
        kp = 0.1; ki = 0.00; kd = 0.001 # decent on straight Average deviation: 1.096
    elif "Rturn" in topo_id:
        kp = 0.8125; ki = 0.00; kd = 0.3
    elif "Lturn" in topo_id:
        kp = 0.5; ki = 0.00; kd = 0.3
    else:
        kp = 0.75; ki = 0.001; kd = 0.2  # decent
        # kp = 1.5; ki = 0.01; kd = 0.267
        # kp = 0.8125; ki = 0.00; kd = 0.3
    error = steer_setpoint - curr_steering
    deriv = (error - steer_prev_error) / dt
    steer_integral = steer_integral + error * dt
    w = kp * error + ki * steer_integral + kd * deriv
    # print(f"steering_PID({curr_steering=:.3f}  \t{steer_setpoint=:.3f}  \t{dt=:.3f})  \t{steer_prev_error=:.3f}  \t{w=:.3f}")
    steer_prev_error = error
    return w

def get_expert_action(vehicle):
    global centerline_interpolated
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle.state['front'])
    dist = min(distance_from_centerline)
    coming_index = 3
    i = np.where(distance_from_centerline == dist)[0][0]
    next_point = centerline_interpolated[(i + coming_index) % len(centerline_interpolated)]
    # next_point2 = centerline_interpolated[(i + coming_index*2) % len(centerline_interpolated)]
    theta = angle_between(vehicle.state, next_point)
    action = theta / (2 * math.pi)
    fig, ax = plt.subplots()
    plt.plot([vehicle.state["front"][0], vehicle.state["pos"][0]], [vehicle.state["front"][1], vehicle.state["pos"][1]], label="car")
    plt.plot([j[0] for j in centerline_interpolated[i+coming_index:i+20]], [j[1] for j in centerline_interpolated[i+coming_index:i+20]], label="centerline")
    plt.plot(next_point[0], next_point[1], 'ro', label="next waypoint")
    plt.legend()
    plt.title(f"{action=:.3f} theta={math.degrees(theta):.1f}")
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("get_expert_action(vehicle)", img)
    cv2.waitKey(1)
    plt.close('all')
    return action, theta

def angle_between(vehicle_state, next_waypoint, next_waypoint2=None):
    # vehicle_angle = math.atan2(vehicle_state['front'][1]-vehicle_state['pos'][1], vehicle_state['front'][0]-vehicle_state['pos'][0])
    vehicle_angle = math.atan2(vehicle_state['front'][1] - vehicle_state['pos'][1], vehicle_state['front'][0] - vehicle_state['pos'][0])
    if next_waypoint2 is not None:
        waypoint_angle = math.atan2((next_waypoint2[1] - next_waypoint[1]),(next_waypoint2[0] - next_waypoint[0]))
    else:
        waypoint_angle = math.atan2((next_waypoint[1]-vehicle_state['front'][1]), (next_waypoint[0]-vehicle_state['front'][0]))
    inner_angle = vehicle_angle - waypoint_angle
    return math.atan2(math.sin(inner_angle), math.cos(inner_angle))

''' track ~12.50m wide; car ~1.85m wide '''
def has_car_left_track(vehicle):
    global centerline_interpolated
    vehicle.update_vehicle()
    vehicle_pos = vehicle.state['front']
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle_pos)
    dist = min(distance_from_centerline)
    i = np.where(distance_from_centerline == dist)[0][0]
    leftrightcenter = get_position_relative_to_centerline(vehicle.state['front'], dist, i, centerdist=0.25)
    # print(f"{leftrightcenter=}  \tdist from ctrline={dist:.3f}")
    segment_shape, theta_deg = get_current_segment_shape(vehicle_pos)
    return dist > 4.0, dist, leftrightcenter, segment_shape, theta_deg

'''returns centered=0, left of centerline=-1, right of centerline=1'''
def get_position_relative_to_centerline(front, dist, i, centerdist=1):
    global centerline_interpolated
    A = centerline_interpolated[(i + 1) % len(centerline_interpolated)]
    B = centerline_interpolated[(i + 4) % len(centerline_interpolated)]
    P = front
    d = (P[0]-A[0])*(B[1]-A[1])-(P[1]-A[1])*(B[0]-A[0])
    if abs(dist) < centerdist:
        return 0 # on centerline
    elif d < 0:
        return -1 # left of centerline
    elif d > 0:
        return 1 # right of centerline

def get_current_segment_shape(vehicle_pos):
    global centerline
    distance_from_centerline = dist_from_line(centerline, vehicle_pos)
    dist = min(distance_from_centerline)
    i = np.where(distance_from_centerline == dist)[0][0]
    A = np.array(centerline[(i + 2) % len(centerline)])
    B = np.array(centerline[i])
    C = np.array(roadright[i])
    theta = math.acos(np.vdot(B-A, B-C) / (np.linalg.norm(B-A) * np.linalg.norm(B-C)))
    theta_deg = math.degrees(theta)
    if theta_deg > 110:
        return 1, theta_deg
    elif theta_deg < 70:
        return 2, theta_deg
    else:
        return 0, theta_deg

def main(topo_name="Lturn_uphill"):
    global interventions, episode_steps, centerline
    global steer_integral, steer_prev_error, topo_id
    model_name = "../weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_name, map_location=device).eval()
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("parameters:", pytorch_total_params, "trainable parameters:", pytorch_trainable_params)
    topo_id = topo_name
    transf_id = "fisheye"
    runs = 1
    default_scenario, road_id, seg, reverse = get_topo(topo_id)
    img_dims, fov, transf = get_transf(transf_id)

    vehicle, bng, scenario = setup_beamng(default_scenario=default_scenario, road_id=road_id, transf=transf, reverse=reverse, seg=seg, img_dims=img_dims, fov=fov, vehicle_model='hopper',
                                          beamnginstance='C:/Users/Meriel/Documents/BeamNG.researchINSTANCE3', port=64556)
    distances, deviations, all_episode_steps, interventions, all_inputs, trajectories = [], [], [], [], [], []

    for i in range(runs):
        results = run_scenario(vehicle, bng, scenario, model, default_scenario=default_scenario, road_id=road_id, transf=transf, vehicle_model='hopper', run_number=i, seg=seg)
        print(f"ACTIONS:\n\t{min(results['inputs'])=}"
              f"\n\t{max(results['inputs'])=}"
              f"\n\t{np.mean(results['inputs'])=}"
              f"\n\t{np.median(results['inputs'])=}"
              f"\n\t{np.std(results['inputs'])=}")
        results['distance'] = get_distance_traveled(results['traj'])
        # plot_trajectory(results['traj'], f"{default_scenario}-{model._get_name()}-{road_id}-runtime{results['runtime']:.2f}-dist{results['distance']:.2f}")
        print(f"\nEVALUATOR + BASE MODEL + NEW CAMERA + INV TRANSF, RUN {i}:"
              f"\n\tdistance={results['distance']:.3f}"
              f"\n\tavg dist from center={results['deviation']['mean']:.3f}"
              f"\n\tintervention rate:{(results['interventions'] / results['episode_steps']):.3f}")
        all_inputs.extend(results['inputs'])
        distances.append(results['distance'])
        deviations.append(results['deviation']['mean'])
        interventions.append(results['interventions'])
        all_episode_steps.append(results['episode_steps'])
        trajectories.append(results["traj"])
        steer_integral, steer_prev_error = 0.0, 0.0
    print(f"OUT OF {runs} RUNS:"
          f"\n\tAverage distance: {(sum(distances)/len(distances)):.1f}"
          f"\n\tAverage deviation: {(sum(deviations) / len(deviations)):.3f}"
          f"\n\tAverage intervention rate:{(sum(interventions) / sum(all_episode_steps)):.3f}"
          f"\n\t{distances=}"
          f"\n\t{deviations=}"
          f"\n\t{interventions=}"
          f"\n\t{all_episode_steps=}"
          f"\n\t{min(all_inputs)=:.3f}"
          f"\n\t{max(all_inputs)=:.3f}"
          f"\n\t{np.mean(all_inputs)=:.3f}"
          f"\n\t{np.std(all_inputs)=:.3f}")
    id = "basemodel+invtransf+0.05evalcorr" # "evalalone" #
    # try:
    # plot_deviation(trajectories, "DAVE2V3", ".", savefile=f"{topo_id}-{transf_id}-{id}")
    # except:
    #     plot_deviation(trajectories, "DAVE2V3", ".", savefile=f"{topo_id}-{transf_id}-{id}")
    bng.close()
    time.sleep(10)


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    # DONE: "extra_small_islandcoast_a_nw","extra_jungleouter_road_a", "extra_jungledrift_road_d", "extra_westcoastrocks", "extra_jungleouter_road_b"
    # topos = ["Lturn_uphill", "extra_dock","countryrd", "Rturn_mtnrd",  "Rturn", "Lturn", "extra_utahlong",
    #             "extra_utahlong2", "extra_utahexittunnel", "extra_utahswitchback", "Rturn_small_island_ai_1", "Rturn_int_a_small_island",
    #             "extra_junglemountain_road_c",  "Rturn_industrialnarrowservicerd", "Rturnrockylinedmtnroad",
    #             "extra_dock", "Rturn_maintenancerd", "Rturn_narrowcutthru", "Rturn_bigshoulder", "Rturn_servicecutthru",
    #             "extrawinding_industrialrcasphalta", "extrawinding_industrial7978","Rturn_hirochitrack", "Rturn_sidequest", "Rturn_lanelines",
    #             "Rturn_bridge", "Lturn_narrowservice", "Rturn_industrialrc_asphaltd", "Rturn_industrial7978",
    #             "Rturn_industrialrc_asphaltb", "Lturn_junglemountain_road_e", "extra_jungledrift_road_b", "extra_jungle8161",
    #             "extra_junglemountain_alt_f", "extra_junglemountain_road_i", "extra_junglemeander8114", "extra_jungledrift_road_m",
    #             "extra_jungledrift_road_k", "extra_jungle8131", "extra_junglemountain_alt_a", "extra_junglemeander7994", "extra_jungle8000",
    #             "extra_dock", "extra_winding",  "extra_whatever", "extra_utahtunnel", "extra_wideclosedtrack",
    #             "extra_wideclosedtrack2", "extra_windingnarrowtrack", "extra_windingtrack", "extra_multilanehighway", "extra_multilanehighway2",
    #             "extra_jungleouter_road_c", "extrawinding_industrialtrack", "straight",
    #             "Lturn_test3", "extra_driver_trainingvalidation2", "extra_lefthandperimeter",
    #             "narrowjungleroad1", "narrowjungleroad2", "Lturnyellow", "straightcommercialroad", "Rturninnertrack", "straightwidehighway",
    #             "Rturncommercialunderpass", "Lturncommercialcomplex", "Lturnpasswarehouse", "Rturnserviceroad", "Rturnlinedmtnroad",
    #             "Rturn_industrial8022whitepave", "Rturn_industrial8068widewhitepave", "Rturn_industrialrc_asphaltc",
    #             "extra_westdockleftside", "extra_westmtnroad", "extra_jungledrift_road_f", "extra_junglemain_tunnel", "extra_jungledrift_road_s",
    #             "extra_jungledrift_road_e", "extra_jungledrift_road_a", "extra_junglemountain_road_h",
    #             "dealwithlater"]
    # FIX: , "extrawinding_industrial8067","extra_windyjungle8082",
    # RUN:
    # todo:
    topos = ["extra_westoutskirts", "extra_westsuburbs", "extra_westsuburbs", "extra_westunderpasses", "extra_westLturnway", "extra_westofframp",]
    #topos = ["extra_test7", "Lturn_uphill", "extra_test2", "extra_test1",  "extra_test3", "extra_test1", "extra_test4",]
    for t in topos:
        print(f"\n\nCOLLECTION FOR TOPO: {t}")
        main(topo_name=t)