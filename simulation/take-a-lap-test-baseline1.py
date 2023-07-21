import pickle
import os.path
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import logging
import scipy.misc
import copy
import torch
import statistics, math
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
from pathlib import Path
import csv
from ast import literal_eval
import PIL
import sys
# sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/DAVE2-Keras')
# sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/superdeepbillboard')
# sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/BeamNGpy')
# sys.path.append(f'/mnt/c/Users/Meriel/Documents/GitHub/BeamNGpy/src/')
# sys.path.append(f'{args.path2src}/GitHub/superdeepbillboard')
# sys.path.append(f'{args.path2src}/GitHub/BeamNGpy')
# sys.path.append(f'{args.path2src}/GitHub/BeamNGpy/src/')
# from wand.image import Image as WandImage
import DAVE2pytorch
from DAVE2pytorch import DAVE2PytorchModel, DAVE2v3
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
from beamngpy import ProceduralCube

from torchvision.transforms import Compose, ToPILImage, ToTensor
from models import vqvae
from models.vqvae import VQVAE
import string
from sim_utils import *
from transformations import transformations
from transformations import detransforms
# globals
integral, prev_error = 0.0, 0.0
overall_throttle_setpoint = 40
setpoint = overall_throttle_setpoint
lanewidth = 3.75 #2.25
centerline = []
centerline_interpolated = []
roadleft = []
roadright = []


def setup_sensors(vehicle, img_dims, fov=51):
    fov = fov # 60 works for full lap #63 breaks on hairpin turn
    resolution = img_dims #(240, 135) #(400,225) #(320, 180) #(1280,960) #(512, 512)
    pos = (-0.5, 0.38, 1.3)
    direction = (0, 1.0, 0)
    front_camera = Camera(pos, direction, fov, resolution,
                          colour=True, depth=True, annotation=True)

    vehicle.attach_sensor('front_cam', front_camera)
    vehicle.attach_sensor('gforces', GForces())
    vehicle.attach_sensor('electrics', Electrics())
    vehicle.attach_sensor('damage', Damage())
    vehicle.attach_sensor('timer', Timer())
    return vehicle


def ms_to_kph(wheelspeed):
    return wheelspeed * 3.6


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


def diff_damage(damage, damage_prev):
    if damage is None or damage_prev is None:
        return 0
    else:
        return damage['damage'] - damage_prev['damage']


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
        plt.plot(x, y, label="Run {}".format(i), alpha=0.75)

    x.sort()
    y.sort()
    min_x, max_x = x[0], x[-1]
    min_y, max_y = y[0], y[-1]
    plt.xlim(min_x - 12, max_x + 12)
    plt.ylim(min_y - 12, max_y + 12)

    plt.title(f'Trajectories with {model} \n{savefile}')
    # plt.legend()
    plt.legend(loc=2, prop={'size': 6})
    plt.draw()
    print(f"Saving image to {deflation_pattern}/{savefile}.jpg")
    plt.savefig(f"{deflation_pattern}/{savefile}.jpg")
    plt.close()
    # plt.show()
    # plt.pause(0.1)


def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment
    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892
    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)


#return distance between two any-dimenisonal points
def distance(a, b):
    sqr = sum([math.pow(ai-bi, 2) for ai, bi in zip(a,b)])
    # return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
    return math.sqrt(sqr)


def dist_from_line(centerline, point):
    a = [[x[0],x[1]] for x in centerline[:-1]]
    b = [[x[0],x[1]] for x in centerline[1:]]
    a = np.array(a)
    b = np.array(b)
    dist = lineseg_dists([point[0], point[1]], a, b)
    return dist


def calc_deviation_from_center(centerline, traj):
    dists = []
    for point in traj:
        dist = dist_from_line(centerline, point)
        dists.append(min(dist))
    stddev = statistics.stdev(dists)
    avg = sum(dists) / len(dists)
    return {"stddev":stddev, "mean":avg}


def plot_racetrack_roads(roads, bng, default_scenario, road_id, reverse=False):
    print("Plotting scenario roads...")
    sp = spawn_point(default_scenario, road_id, reverse=reverse)
    colors = ['b','g','r','c','m','y','k']
    symbs = ['-','--','-.',':','.',',','v','o','1',]
    selectedroads = []
    for road in roads:
        road_edges = bng.get_road_edges(road)
        x_temp = []
        y_temp = []
        add = True
        xy_def = [edge['middle'][:2] for edge in road_edges]
        dists = [distance(xy_def[i], xy_def[i+1]) for i,p in enumerate(xy_def[:-1])]
        s = sum(dists)
        if (s < 200):
            continue
        for edge in road_edges:
            # if edge['middle'][0] < -250 or edge['middle'][0] > 50 or edge['middle'][1] < 0 or edge['middle'][1] > 300:
            if edge['middle'][1] < -50 or edge['middle'][1] > 250:
                add = False
                break
            if add:
                x_temp.append(edge['middle'][0])
                y_temp.append(edge['middle'][1])
        if add:
            symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
            plt.plot(x_temp, y_temp, symb, label=road)
            selectedroads.append(road)
    for r in selectedroads: # ["8179", "8248", "8357", "8185", "7770", "7905", "8205", "8353", "8287", "7800", "8341", "7998"]:
        a = bng.get_road_edges(r)
        print(r, a[0]['middle'])
    plt.plot([sp['pos'][0]], [sp['pos'][1]], "bo")
    plt.legend()
    plt.show()
    plt.pause(0.001)


def road_analysis_old(bng, road_id):
    global centerline, roadleft, roadright
    # plot_racetrack_roads(bng.get_roads(), bng)
    print(f"Getting road {road_id}...")
    edges = bng.get_road_edges(road_id)
    actual_middle = [edge['middle'] for edge in edges]
    roadleft = [edge['left'] for edge in edges]
    roadright = [edge['right'] for edge in edges]
    adjusted_middle = [np.array(edge['middle']) + (np.array(edge['left']) - np.array(edge['middle']))/4.0 for edge in edges]
    centerline = actual_middle
    return actual_middle, adjusted_middle


def road_analysis(bng, road_id):
    global centerline, roadleft, roadright
    print("Performing road analysis...")
    # get_nearby_racetrack_roads(point_of_in=(-391.0,-798.8, 139.7))
    # self.plot_racetrack_roads()
    print(f"Getting road {road_id}...")
    edges = bng.get_road_edges(road_id)
    centerline = [edge['middle'] for edge in edges]
    # self.roadleft = [edge['left'] for edge in edges]
    # self.roadright = [edge['right'] for edge in edges]
    if road_id == "8185":
        edges = bng.get_road_edges("8096")
        roadleft = [edge['middle'] for edge in edges]
        edges = bng.get_road_edges("7878")  # 7820, 7878, 7805
        roadright = [edge['middle'] for edge in edges]
    else:
        roadleft = [edge['left'] for edge in edges]
        roadright = [edge['right'] for edge in edges]
    return centerline, centerline


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


def create_ai_line_from_road_with_interpolation(spawn, bng, road_id):
    global centerline, centerline_interpolated
    line = []; points = []; point_colors = []; spheres = []; sphere_colors = []; traj = []
    print("Performing road analysis...")
    actual_middle, adjusted_middle = road_analysis(bng, road_id)
    # plt.plot([i[0] for i in actual_middle], [i[1] for i in actual_middle])
    # plt.show()
    print(f"{actual_middle[0]=}, {actual_middle[-1]=}")
    # middle_end = adjusted_middle[:3]
    # middle = adjusted_middle[3:]
    # temp = [list(spawn['pos'])]; temp.extend(middle); middle = temp
    # middle.extend(middle_end)
    middle = actual_middle
    timestep = 0.1; elapsed_time = 0; count = 0
    # set up adjusted centerline
    for i,p in enumerate(middle[:-1]):
        # interpolate at 1m distance
        if distance(p, middle[i+1]) > 1:
            y_interp = interpolate.interp1d([p[0], middle[i+1][0]], [p[1], middle[i+1][1]])
            num = int(distance(p, middle[i+1]))
            xs = np.linspace(p[0], middle[i+1][0], num=num, endpoint=True)
            ys = y_interp(xs)
            for x,y in zip(xs,ys):
                traj.append([x,y])
        else:
            elapsed_time += distance(p, middle[i+1]) / 12
            traj.append([p[0],p[1]])
            linedict = {"x": p[0], "y": p[1], "z": p[2], "t": elapsed_time}
            line.append(linedict)
            count += 1
    # set up debug line
    for i,p in enumerate(actual_middle[:-1]):
        points.append([p[0], p[1], p[2]])
        point_colors.append([0, 1, 0, 0.1])
        spheres.append([p[0], p[1], p[2], 0.25])
        sphere_colors.append([1, 0, 0, 0.8])
        count += 1
    print("spawn point:{}".format(spawn))
    print("beginning of script:{}".format(middle[0]))
    # plot_trajectory(traj, "Points on Script (Final)", "AI debug line")
    # centerline = copy.deepcopy(traj)
    centerline_interpolated = copy.deepcopy(traj)
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return line, bng


# track is approximately 12.50m wide
# car is approximately 1.85m wide
def has_car_left_track(vehicle_pos, max_dist=5.0):
    global centerline_interpolated
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle_pos)
    dist = min(distance_from_centerline)
    return dist > max_dist, dist


def setup_beamng(default_scenario, road_id, reverse=False, seg=1, img_dims=(240,135), fov=51, vehicle_model='etk800', default_color="green", steps_per_sec=15,
                 beamnginstance='C:/Users/Meriel/Documents/BeamNG.researchINSTANCE4', port=64956):
    global base_filename

    random.seed(1703)
    setup_logging()
    print(road_id)
    beamng = BeamNGpy('localhost', port, home='C:/Users/Meriel/Documents/BeamNG.research.v1.7.0.1', user=beamnginstance)
    # beamng = BeamNGpy('localhost', 64256, home='C:/Users/Meriel/Documents/BeamNG.tech.v0.21.3.0', user='C:/Users/Meriel/Documents/BeamNG.tech')
    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model, licence='EGO', color=default_color)
    print(f"IMG DIMS IN SETUP={img_dims}")
    vehicle = setup_sensors(vehicle, img_dims, fov=fov)
    spawn = spawn_point(default_scenario, road_id, reverse=reverse, seg=seg)
    print(default_scenario, road_id, seg, spawn)
    scenario.add_vehicle(vehicle, pos=spawn['pos'], rot=None, rot_quat=spawn['rot_quat'])
    try:
        add_barriers(scenario, default_scenario)
    except FileNotFoundError as e:
        print(e)
    print(road_id)
    scenario.make(beamng)
    bng = beamng.open(launch=True)
    bng.set_deterministic()
    bng.set_steps_per_second(steps_per_sec)
    bng.load_scenario(scenario)
    bng.start_scenario()
    ai_line, bng = create_ai_line_from_road_with_interpolation(spawn, bng, road_id)
    bng.pause()
    assert vehicle.skt
    # bng.resume()
    return vehicle, bng, scenario


def run_scenario(vehicle, bng, scenario, model, default_scenario, road_id, reverse=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), seg=None, vqvae=None,
                 transf=None, detransf=None):
    global base_filename
    global integral, prev_error, setpoint
    spawn = spawn_point(default_scenario, road_id, reverse=reverse, seg=seg)
    cutoff_point = spawn['pos']
    integral = 0.0
    prev_error = 0.0
    bng.restart_scenario()
    plt.pause(0.01)

    # perturb vehicle
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)
    image = sensors['front_cam']['colour'].convert('RGB')
    pitch = vehicle.state['pitch'][0]
    roll = vehicle.state['roll'][0]
    z = vehicle.state['pos'][2]
    spawn = spawn_point(default_scenario, road_id, reverse=reverse, seg=seg)

    wheelspeed = kph = throttle = runtime = distance_from_center = 0.0
    prev_error = setpoint
    kphs = []; traj = []; steering_inputs = []; throttle_inputs = []; timestamps = []
    damage = 0.0
    final_img = None
    total_loops = total_imgs = total_predictions = 0
    start_time = sensors['timer']['time']
    outside_track = False
    transform = Compose([ToTensor()])
    while kph < 35:
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        vehicle.control(throttle=1., steering=0., brake=0.0)
        bng.step(1, wait=True)
    while damage <= 1:
        # collect images
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        image = sensors['front_cam']['colour'].convert('RGB')
        image_seg = sensors['front_cam']['annotation'].convert('RGB')
        image_depth = sensors['front_cam']['depth'].convert('RGB')
        if transf is not None:
            if "depth" in transf:
                transformations.blur_with_depth_image(np.array(image), np.array(image_depth))

        if detransf is not None:
            if "res" in detransf:
                image = image.resize((192, 108))
                # image = cv2.resize(np.array(image), (135,240))
            elif "fisheye" in detransf:
                image = detransforms.defisheye(np.array(image))
        cv2.imshow('car view', np.array(image)[:, :, ::-1])
        cv2.waitKey(1)
        total_imgs += 1
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        dt = (sensors['timer']['time'] - start_time) - runtime
        try:
            processed_img = model.process_image(image).to(device)
        except Exception as e:
            processed_img = transform(np.asarray(image))[None]
        prediction = model(processed_img.type(torch.float32))
        steering = float(prediction.item())
        runtime = sensors['timer']['time'] - start_time
        total_predictions += 1
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
        vehicle.update_vehicle()
        traj.append(vehicle.state['pos'])

        kphs.append(ms_to_kph(wheelspeed))
        total_loops += 1
        final_img = image
        dists = dist_from_line(centerline, vehicle.state['pos'])
        m = np.where(dists == min(dists))[0][0]
        if damage > 1.0:
            print(f"Damage={damage:.3f}, exiting...")
            print(f"Try next spawn: {centerline[m+5]}")
            break
        bng.step(1, wait=True)

        if distance2D(vehicle.state["pos"], cutoff_point) < 12 and len(traj) > 30:
            print("Reached cutoff point, exiting...")
            break

        outside_track, distance_from_center = has_car_left_track(vehicle.state['pos'], max_dist=6.0)
        if outside_track:
            print("Left track, exiting...")
            break

    cv2.destroyAllWindows()

    deviation = calc_deviation_from_center(centerline, traj)
    results = {'runtime': round(runtime,3), 'damage': damage, 'kphs':kphs, 'traj':traj, 'pitch': round(pitch,3),
               'roll':round(roll,3), "z":round(z,3), 'final_img':final_img, 'deviation':deviation
               }
    return results


def distance2D(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_distance_traveled(traj):
    dist = 0.0
    for i in range(len(traj[:-1])):
        dist += math.sqrt(math.pow(traj[i][0] - traj[i+1][0],2) + math.pow(traj[i][1] - traj[i+1][1],2) + math.pow(traj[i][2] - traj[i+1][2],2))
    return dist


def turn_X_degrees(rot_quat, degrees=90):
    r = R.from_quat(list(rot_quat))
    r = r.as_euler('xyz', degrees=True)
    r[2] = r[2] + degrees
    r = R.from_euler('xyz', r, degrees=True)
    return tuple(r.as_quat())


def add_barriers(scenario, default_scenario):
    with open(f'posefiles/{default_scenario}_barrier_locations.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            pos = line[0].split(',')
            pos = tuple([float(i) for i in pos])
            rot_quat = line[1].split(',')
            rot_quat = tuple([float(j) for j in rot_quat])
            rot_quat = turn_X_degrees(rot_quat, degrees=-115)
            ramp = StaticObject(name='barrier{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(1, 1, 1),
                                shape='levels/Industrial/art/shapes/misc/concrete_road_barrier_a.dae')
            scenario.add_object(ramp)

# def fisheye_wand(image, filename=None):
#     with WandImage.from_array(image) as img:
#         img.virtual_pixel = 'transparent'
#         img.distort('barrel', (0.1, 0.0, -0.05, 1.0))
#         img.alpha_channel = False
#         img = np.array(img, dtype='uint8')
#         return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
#
# def fisheye_inv(image):
#     with WandImage.from_array(image) as img:
#         img.virtual_pixel = 'transparent'
#         img.distort('barrel_inverse', (0.0, 0.0, -0.5, 1.5))
#         img = np.array(img, dtype='uint8')
#         return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def zero_globals():
    global centerline, centerline_interpolated, roadleft, roadright
    centerline = []
    centerline_interpolated = []
    roadleft = []
    roadright = []


def main(topo_id, hash="000"):
    global base_filename
    zero_globals()
    model_name = "F:/dave2-base-models/DAVE2v3-108x192-145samples-5000epoch-5364842-7_4-17_15-XACCPQ-140EPOCHS/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch126-best044.pt"
    detransf_id = "resdec2"
    transf_id = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_name, map_location=device).eval()
    vqvae_name = None
    vqvae = None
    vqvae_id = "baseline1"
    default_scenario, road_id, seg, reverse = get_topo(topo_id)
    img_dims, fov, transf = get_transf(detransf_id)
    print(f"IMAGE DIMS={img_dims}")
    vehicle, bng, scenario = setup_beamng(default_scenario=default_scenario, road_id=road_id, seg=seg, reverse=reverse, img_dims=img_dims, fov=fov, vehicle_model='hopper',
                                          beamnginstance='C:/Users/Meriel/Documents/BeamNG.researchINSTANCE3', port=64156)
    distances, deviations, trajectories = [], [], []
    runs = 5

    filepathroot = f"{'/'.join(model_name.split('/')[:-1])}/{vqvae_id}-{detransf_id}-{default_scenario}-{road_id}-{topo_id}topo-{runs}runs-{hash}/"
    print(f"{filepathroot=}")
    Path(filepathroot).mkdir(exist_ok=True, parents=True)

    for i in range(runs):
        results = run_scenario(vehicle, bng, scenario, model, default_scenario=default_scenario, road_id=road_id, seg=seg, vqvae=vqvae, transf=transf_id, detransf=detransf_id)
        results['distance'] = get_distance_traveled(results['traj'])
        # plot_trajectory(results['traj'], f"{default_scenario}-{model._get_name()}-{road_id}-runtime{results['runtime']:.2f}-dist{results['distance']:.2f}")
        print(f"\nBASE MODEL USING IMG DIMS {img_dims} RUN {i}:"
              f"\n\tdistance={results['distance']:1f}"
              f"\n\tavg dist from center={results['deviation']['mean']:3f}")
        distances.append(results['distance'])
        deviations.append(results['deviation']['mean'])
        trajectories.append(results["traj"])
    summary = {
        "trajectories": trajectories,
        "dists_from_centerline": deviations,
        "dists_travelled": distances,
        "img_dims": img_dims,
        "centerline_interpolated": centerline_interpolated,
        "roadleft": roadleft,
        "roadright": roadright,
        "default_scenario": default_scenario,
        "road_id": road_id,
        "topo_id": topo_id,
        "transf_id": transf_id,
        "vqvae_name": vqvae_name,
        "model_name": model_name,
        # "obs_shape": self.obs_shape,
        # "action_space": self.action_space,
        # "wall_clock_time": time.time() - self.start_time,
        # "sim_time": self.runtime
    }

    picklefile = open(f"{filepathroot}/summary-{model_name.split('/')[-1]}_{vqvae_id}.pickle", 'wb')
    pickle.dump(summary, picklefile)
    picklefile.close()
    print(f"{topo_id} OUT OF {runs} RUNS:\n\tAvg. distance: {(sum(distances)/len(distances)):.1f}"
          f"\n\tAvg. deviation: {(sum(deviations) / len(deviations)):.3f}"
          f"\n\t{distances=}"
          f"\n\t{deviations:}"
          f"\n\t{vqvae_name=}"
          f"\n\t{model_name=}")
    id = f"basemodelalone-{vqvae_id}"
    try:
        plot_deviation(trajectories, "DAVE2V3 ", filepathroot, savefile=f"{topo_id}-{transf_id}-{id}")
    except:
        plot_deviation(trajectories, "DAVE2V3", filepathroot, savefile=f"{topo_id}-{transf_id}-{id}")
    bng.close()


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    hash = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    main("Rturnserviceroad", hash=hash)
    main("extra_windingnarrowtrack", hash=hash)
    main("extra_windingtrack", hash=hash)
    main("Rturn_bigshouldertopo", hash=hash)
    main("Rturn_bridgetopo", hash=hash)