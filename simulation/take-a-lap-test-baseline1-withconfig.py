import sys
sys.path.append("C:/Users/Meriel/Documents/GitHub/BeamNGpy/src")
sys.path.append("C:/Users/Meriel/Documents/GitHub/IFAN")
sys.path.append("C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation")
from pathlib import Path
import string
import pickle
import PIL
import cv2
import random
import numpy as np
import logging
import copy
from scipy import interpolate
import torch
from torchvision.transforms import Compose, ToPILImage, ToTensor
import argparse
import pandas as pd

# from ast import literal_eval
# from wand.image import Image as WandImage
import DAVE2pytorch
from DAVE2pytorch import DAVE2PytorchModel, DAVE2v3
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
from sim_utils import *
from transformations import transformations
from transformations import detransformations

# globals
integral, prev_error = 0.0, 0.0
overall_throttle_setpoint = 40
setpoint = overall_throttle_setpoint
lanewidth = 3.75 #2.25
centerline = []
centerline_interpolated = []
roadleft = []
roadright = []


def throttle_PID(kph, dt, steering=0.):
    global integral, prev_error, setpoint
    kp = 0.19; ki = 0.0001; kd = 0.008
    if abs(steering) > 0.15:
        setpoint = 30
    else:
        setpoint = 40
    error = setpoint - kph
    if dt > 0:
        deriv = (error - prev_error) / dt
    else:
        deriv = 0
    integral = integral + error * dt
    w = kp * error + ki * integral + kd * deriv
    prev_error = error
    return w


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


def road_analysis(bng, default_scenario, road_id, spawn_pt=None):
    global centerline, roadleft, roadright
    print("Performing road analysis...")
    # get_nearby_racetrack_roads(point_of_in=(-391.0,-798.8, 139.7))
    # plot_racetrack_roads(bng.get_roads(), bng, default_scenario, road_id, reverse=False, sp={"pos":spawn_pt})
    print(f"Getting road {road_id}...")
    edges = bng.get_road_edges(road_id)
    centerline = [edge['middle'] for edge in edges]
    if road_id == "8185":
        edges = bng.get_road_edges("8096")
        roadleft = [edge['middle'] for edge in edges]
        edges = bng.get_road_edges("7878")  # 7820, 7878, 7805
        roadright = [edge['middle'] for edge in edges]
    else:
        roadleft = [edge['left'] for edge in edges]
        roadright = [edge['right'] for edge in edges]
    return centerline, centerline


def create_ai_line_from_road_with_interpolation(spawn, bng, default_scenario, road_id):
    global centerline, centerline_interpolated
    line, points, point_colors, spheres, sphere_colors, traj = [], [], [], [], [], []
    print("Performing road analysis...")
    actual_middle, adjusted_middle = road_analysis(bng, default_scenario, road_id, spawn)
    middle = actual_middle
    count = 0
    # set up adjusted centerline
    for i,p in enumerate(middle[:-1]):
        # interpolate at 1m distance
        if distance(p, middle[i+1]) > 1:
            y_interp = interpolate.interp1d([p[0], middle[i+1][0]], [p[1], middle[i+1][1]])
            num = int(distance(p, middle[i+1]))
            xs = np.linspace(p[0], middle[i+1][0], num=num, endpoint=True)
            ys = y_interp(xs)
            for x, y in zip(xs, ys):
                traj.append([x, y])
        else:
            traj.append([p[0], p[1]])
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
    print(f"{centerline_interpolated[0]=}")
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return line, bng


def setup_beamng(default_scenario, spawn_pos, rot_quat,  road_id, reverse=False, seg=1, img_dims=(240,135), fov=51, vehicle_model='etk800', default_color="green", steps_per_sec=15,
                 beamnginstance='C:/Users/Meriel/Documents/BeamNG.researchINSTANCE4', port=64956):
    global base_filename

    # random.seed(1703)
    setup_logging()
    beamng = BeamNGpy('localhost', port, home='F:/BeamNG.research.v1.7.0.1', user=beamnginstance)
    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model, licence='EGO', color=default_color)
    vehicle = setup_sensors(vehicle, img_dims, fov=fov)
    # spawn = spawn_point(default_scenario, road_id, reverse=reverse, seg=seg)
    print(default_scenario, road_id, seg,  spawn_pos, rot_quat, )
    scenario.add_vehicle(vehicle, pos=spawn_pos, rot=None, rot_quat=rot_quat)
    try:
        add_barriers(scenario, default_scenario)
    except FileNotFoundError as e:
        print(e)
    scenario.make(beamng)
    bng = beamng.open(launch=True)
    bng.set_deterministic()
    bng.set_steps_per_second(steps_per_sec)
    bng.load_scenario(scenario)
    bng.start_scenario()
    ai_line, bng = create_ai_line_from_road_with_interpolation(spawn_pos, bng, default_scenario, road_id)
    bng.pause()
    assert vehicle.skt
    return vehicle, bng, scenario


def run_scenario(vehicle, bng, scenario, model, default_scenario, road_id, reverse=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), seg=None, vqvae=None,
                 detransf=None, topo=None, cuton_pt=None, cutoff_pt=None):
    global base_filename, centerline_interpolated
    global integral, prev_error, setpoint
    spawn = spawn_point(default_scenario, road_id, reverse=reverse, seg=seg)
    if cutoff_pt is None:
        cutoff_pt = centerline[-1]
    integral = 0.0
    prev_error = 0.0
    bng.restart_scenario()
    plt.pause(0.01)

    # perturb vehicle
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)
    image = sensors['front_cam']['colour'].convert('RGB')
    # image.save(f"./start-{topo}-{detransf}.jpg")
    spawn = spawn_point(default_scenario, road_id, reverse=reverse, seg=seg)

    wheelspeed = kph = throttle = runtime = distance_from_center = 0.0
    prev_error = setpoint
    kphs = []; traj = []; steering_inputs = []; throttle_inputs = []; timestamps = []
    damage = 0.0
    final_img = None
    total_loops = total_imgs = total_predictions = 0
    start_time = sensors['timer']['time']
    outside_track = False
    distance_to_cuton = 100
    transform = Compose([ToTensor()])
    last_timestamp = start_time
    while kph < 30 or distance_to_cuton > 3.:
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        dt = sensors['timer']['time'] - last_timestamp
        steering = line_follower(centerline_interpolated, vehicle.state['front'], vehicle.state['pos'], vehicle.state['dir'], topo, vehicle.state, vehicle.get_bbox())
        throttle = throttle_PID(kph, dt, steering=steering)
        vehicle.control(throttle=throttle, steering=steering, brake=0.0)
        last_timestamp = sensors['timer']['time']
        bng.step(1, wait=True)
        vehicle.update_vehicle()
        distance_to_cuton = distance2D(vehicle.state["pos"], cuton_pt)
        print(f"{steering=:.3f} \t{distance_to_cuton=:.1f} {dt=:.3f}")
    while damage <= 1:
        # collect images
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        image = sensors['front_cam']['colour'].convert('RGB')
        image_seg = sensors['front_cam']['annotation'].convert('RGB')
        image_depth = sensors['front_cam']['depth'].convert('RGB')

        if "res" in detransf:
            image = image.resize((192, 108))
            # image = cv2.resize(np.array(image), (135,240))
        elif "fisheye" in detransf:
            image = detransformations.defisheye(np.array(image))
        elif "depth" in detransf:
            image_shallowdof = transformations.blur_with_depth_image(np.array(image), np.array(image_depth))
            image = detransformations.deblur(np.array(image_shallowdof))

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

        throttle = throttle_PID(kph, dt, steering=steering)
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
        dist_to_cutoff = distance2D(vehicle.state["pos"], cutoff_pt)
        # print(f"{dist_to_cutoff=:.1f}")
        if dist_to_cutoff < 12 and len(traj) > 30:
            print("Reached cutoff point, exiting...")
            break

        outside_track, distance_from_center = has_car_left_track(vehicle.state['pos'], centerline_interpolated, max_dist=6.0)
        if outside_track:
            print("Left track, exiting...")
            break

    cv2.destroyAllWindows()

    deviation = calc_deviation_from_center(centerline, traj)
    results = {'runtime': round(runtime,3), 'damage': damage, 'kphs':kphs, 'traj':traj,
               'final_img':final_img, 'deviation':deviation
               }
    return results


def zero_globals():
    global centerline, centerline_interpolated, roadleft, roadright
    centerline = []
    centerline_interpolated = []
    roadleft = []
    roadright = []


def main(topo_id, spawn_pos, rot_quat, cluster="000", hash="000", detransf_id=None, cuton_pt=None, cutoff_pt=None):
    global base_filename
    zero_globals()
    model_name = "../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_name, map_location=device).eval()
    vqvae_name = None
    vqvae = None
    baseline_id = "baseline1"
    default_scenario, road_id, seg, reverse = get_topo(topo_id)
    img_dims, fov, transf = get_transf(detransf_id)
    print(f"IMAGE DIMS={img_dims}")
    vehicle, bng, scenario = setup_beamng(default_scenario, spawn_pos, rot_quat, road_id=road_id, seg=seg, reverse=reverse, img_dims=img_dims, fov=fov, vehicle_model='hopper',
                                          beamnginstance='F:/BeamNG.researchINSTANCE2', port=64416) # port must be 0-65535
    distances, deviations, trajectories, runtimes = [], [], [], []
    runs = 5

    filepathroot = f"simresults/sanitycheckREVISEDCUTONS-{baseline_id}-{detransf_id}-{hash}/{baseline_id}-{detransf_id}-{topo_id}topo-cluster{cluster}-{runs}runs-{hash}/"
    print(f"{filepathroot=}")
    Path(filepathroot).mkdir(exist_ok=True, parents=True)

    point_nearest = get_nearest_point(centerline, spawn_pos)
    print(f"{point_nearest=}")
    if topo_id == "extra_windingtrack" or topo_id == "Rturncommercialunderpass":
        centerline.reverse()
        centerline_interpolated.reverse()
    distances, deviations, deviations_seq, trajectories, runtimes = [], [], [], [], []

    for i in range(runs):
        results = run_scenario(vehicle, bng, scenario, model, default_scenario=default_scenario, road_id=road_id, seg=seg,
                               vqvae=vqvae, detransf=detransf_id, topo=topo_id, cuton_pt=cuton_pt, cutoff_pt=cutoff_pt)
        results['distance'] = get_distance_traveled(results['traj'])
        print(f"\n{baseline_id} {topo_id} {detransf_id} {img_dims=} RUN {i}:"
              f"\n\tdistance={results['distance']:1f}"
              f"\n\tavg dist from center={results['deviation']['mean']:3f}")
        distances.append(results['distance'])
        deviations.append(results['deviation']['mean'])
        trajectories.append(results["traj"])
        runtimes.append(results['runtime'])
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
        "cluster": cluster,
        "detransf_id": detransf_id,
        "vqvae_name": vqvae_name,
        "model_name": model_name,
        "runtimes": runtimes
    }

    picklefile = open(f"{filepathroot}/summary.pickle", 'wb')
    pickle.dump(summary, picklefile)
    picklefile.close()
    print(f"{baseline_id} {topo_id} {detransf_id} OUT OF {runs} RUNS:"
          f"\n\tAvg. distance: {(sum(distances)/len(distances)):.1f}"
          f"\n\tAvg. distance deviation: {np.std(distances):.1f}"
          f"\n\tAvg. center deviation: {(sum(deviations) / len(deviations)):.3f}"
          f"\n\t{distances=}"
          f"\n\t{deviations:}"
          f"\n\t{vqvae_name=}"
          f"\n\t{model_name=}")
    id = f"basemodelalone-{baseline_id}"
    plot_deviation(trajectories, centerline, roadleft, roadright, baseline_id, filepathroot, savefile=f"{topo_id}-{detransf_id}-{id}")
    bng.close()
    return summary


def summarize_results(all_results):
    distances, deviations = [], []
    for result in all_results:
        distances.extend(result["dists_travelled"])
        deviations.extend(result["dists_from_centerline"])
    print(f"5-TRACK SUMMARY:"
          f"\n\tAvg. distance: {(sum(distances)/len(distances)):.1f}"
          f"\n\tAvg. distance deviation: {np.std(distances):.1f}"
          f"\n\tAvg. center deviation: {(sum(deviations) / len(deviations)):.3f}"
          )


def parse_args():
    parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does',
                                     epilog='Text at the bottom of help')
    parser.add_argument('--effect', help='image transformation', default=None)
    args = parser.parse_args()
    print(f"cmd line args:{args}")
    return args


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    # detransf_ids = ["mediumdepth", "resinc", "resdec", "mediumfisheye"]
    df = pd.read_csv("./config-segments_inuse-revised.csv")
    hash = randstr()
    df = df.reset_index()  # make sure indexes pair with number of rows
    random.seed(1703)
    args = parse_args()
    for index, row in df.iterrows():
        config_topo_id = row["TOPOID"]
        spawn_pos = parse_list_from_string(row["SPAWN"])
        rot_quat = parse_list_from_string(row["ROT_QUAT"])
        cluster = row["SEGNUM"]
        cutoff = parse_list_from_string(row["END"])
        cuton_pt = parse_list_from_string(row["START"])
        cutoff_pt = parse_list_from_string(row["CUTOFF"])
        results = main(config_topo_id, spawn_pos, rot_quat, cluster=cluster, hash=hash, detransf_id=args.effect, cuton_pt=cuton_pt, cutoff_pt=cutoff_pt)