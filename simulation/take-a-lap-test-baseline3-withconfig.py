import sys
sys.path.append("C:/Users/Meriel/Documents/GitHub/BeamNGpy/src")
sys.path.append("C:/Users/Meriel/Documents/GitHub/IFAN")
sys.path.append("C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation")
sys.path.append("C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation/DAVE2/")
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

def throttle_PID(kph, dt, steering=None):
    global integral, prev_error, setpoint
    if steering is not None and abs(steering) > 0.15:
        setpoint = 30
    else:
        setpoint = 40
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


def road_analysis(bng, road_id):
    global centerline, roadleft, roadright
    print("Performing road analysis...")
    # get_nearby_racetrack_roads(point_of_in=(-391.0,-798.8, 139.7))
    # self.plot_racetrack_roads()
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


def create_ai_line_from_road_with_interpolation(spawn, bng, road_id):
    global centerline, centerline_interpolated
    line, points, point_colors, spheres, sphere_colors, traj = [], [], [], [], [], []
    print("Performing road analysis...")
    actual_middle, adjusted_middle = road_analysis(bng, road_id)
    print(f"{actual_middle[0]=}, {actual_middle[-1]=}")
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
    centerline_interpolated = copy.deepcopy(traj)
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return line, bng


def setup_beamng(default_scenario, spawn_pos, rot_quat, road_id, reverse=False, seg=1, img_dims=(240,135), fov=51, vehicle_model='etk800', default_color="green", steps_per_sec=15,
                 beamnginstance='F:/BeamNG.researchINSTANCE4', port=64956, topo_id=None):
    global base_filename

    # random.seed(1703)
    setup_logging()
    beamng = BeamNGpy('localhost', port, home='F:/BeamNG.research.v1.7.0.1', user=beamnginstance)
    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model, licence='EGO', color=default_color)
    vehicle = setup_sensors(vehicle, img_dims, fov=fov)
    print(f"{default_scenario=}, {road_id=}, {seg=}, {spawn_pos=}, {rot_quat=}")
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
    ai_line, bng = create_ai_line_from_road_with_interpolation(spawn_pos, bng, road_id)
    if topo_id == "extra_windingtrack" or topo_id == "Rturncommercialunderpass":
        centerline.reverse()
        centerline_interpolated.reverse()
    bng.pause()
    assert vehicle.skt
    return vehicle, bng, scenario


def run_scenario(vehicle, bng, scenario, model, default_scenario, road_id, reverse=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), seg=None, vqvae=None,
                 transf=None, detransf=None, topo=None, cuton_pt=None, cutoff_pt=None):
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
    image.save(f"./start-{topo}-{detransf}.jpg")
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
    distance_to_cuton = 100
    last_timestamp = start_time
    while kph < 30 or distance_to_cuton > 3.5:
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
        print(f"{steering=:.3f} \t{distance_to_cuton=:.1f}")
    while damage <= 1:
        # collect images
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        image = sensors['front_cam']['colour'].convert('RGB')
        image_seg = sensors['front_cam']['annotation'].convert('RGB')
        image_depth = sensors['front_cam']['depth'].convert('RGB')
        if transf is not None:
            if "depth" in transf:
                image = transformations.blur_with_depth_image(np.array(image), np.array(image_depth))
            # elif "res" in transf:
            #     image = image.resize((192, 108))

        cv2.imshow('car view', np.array(image)[:, :, ::-1])
        cv2.waitKey(1)
        total_imgs += 1
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        dt = (sensors['timer']['time'] - start_time) - runtime
        try:
            processed_img = model.process_image(image).to(device)
        except Exception as e:
            processed_img = transform(np.asarray(image))[None]
        # print(f"{processed_img.shape=}")
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
            break
        bng.step(1, wait=True)

        if distance2D(vehicle.state["pos"], cutoff_pt) < 12 and len(traj) > 30:
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


def main(topo_id, spawn_pos, rot_quat, cluster, model_name, hash="000", detransf_id=None, transf_id=None, cuton_pt=None, cutoff_pt=None):
    global base_filename
    zero_globals()

    if transf_id == "resdec":
        from DAVE2resdec import DAVE2v3
    else:
        from DAVE2pytorch import DAVE2PytorchModel, DAVE2v3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_name, map_location=device).eval()
    print(type(model), model.input_shape)
    vqvae_name = None
    vqvae = None
    baseline_id = "baseline3"
    default_scenario, road_id, seg, reverse = get_topo(topo_id)
    img_dims, fov, transf = get_transf(transf_id)
    print(f"TRANSFORM={transf_id} \t IMAGE DIMS={img_dims} \t {spawn_pos=} \t {rot_quat=}")

    vehicle, bng, scenario = setup_beamng(default_scenario, spawn_pos, rot_quat, road_id=road_id, seg=seg, reverse=reverse, img_dims=img_dims, fov=fov, vehicle_model='hopper',
                                          beamnginstance='F:/BeamNG.researchINSTANCE3', port=64356, topo_id=topo_id)
    distances, deviations, trajectories, runtimes = [], [], [], []
    runs = 5
    fpid = model_name.split("/")[-1].replace('.pth', '')
    # filepathroot = f"{'/'.join(model_name.split('/')[:-1])}/{vqvae_id}-{transf_id}-{hash}/{vqvae_id}-{transf_id}-{default_scenario}-{road_id}-{topo_id}topo-{runs}runs-{hash}/"
    filepathroot = f"simresults/B3-REVISEDCUTON-{baseline_id}/{fpid}-{hash}/{baseline_id}-{transf_id}-{topo_id}topo-cluster{cluster}-{runs}runs-{hash}/"

    print(f"{filepathroot=}")
    Path(filepathroot).mkdir(exist_ok=True, parents=True)

    for i in range(runs):
        results = run_scenario(vehicle, bng, scenario, model, default_scenario=default_scenario, road_id=road_id, seg=seg,
                               vqvae=vqvae, transf=transf_id, detransf=detransf_id, topo=topo_id, cuton_pt=cuton_pt, cutoff_pt=cutoff_pt)
        results['distance'] = get_distance_traveled(results['traj'])
        # plot_trajectory(results['traj'], f"{default_scenario}-{model._get_name()}-{road_id}-runtime{results['runtime']:.2f}-dist{results['distance']:.2f}")
        print(f"\nBASE MODEL USING IMG DIMS {img_dims} RUN {i}:"
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
        "transf_id": transf_id,
        "vqvae_name": vqvae_name,
        "model_name": model_name,
        "runtimes": runtimes
    }

    picklefile = open(f"{filepathroot}/summary.pickle", 'wb')
    pickle.dump(summary, picklefile)
    picklefile.close()
    print(f"{topo_id} {transf_id} OUT OF {runs} RUNS:"
          f"\n\tAvg. distance: {(sum(distances)/len(distances)):.1f}"
          f"\n\tAvg. distance deviation: {np.std(distances):.1f}"
          f"\n\tAvg. center deviation: {(sum(deviations) / len(deviations)):.3f}"
          f"\n\t{distances=}"
          f"\n\t{deviations:}"
          f"\n\t{vqvae_name=}"
          f"\n\t{model_name=}")
    id = f"DAVE2V3-{baseline_id}"
    try:
        plot_deviation(trajectories, centerline, roadleft, roadright, "DAVE2V3 ", filepathroot, savefile=f"{topo_id}-{transf_id}-{id}")
    except:
        plot_deviation(trajectories, centerline, roadleft, roadright, "DAVE2V3", filepathroot, savefile=f"{topo_id}-{transf_id}-{id}")
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


def get_model_name(transf_id):
    if transf_id == "mediumfisheye":
        # return "../weights/baseline3/baseline3-fisheye-model-DAVE2v3-108x192-2296epoch-64batch-50Ksamples-epoch2180-best098.pt"
        return "../weights/baseline3/portal409517_baseline3_Tset50K_fisheye-DAVE2v3-108x192-5000epoch-64batch-50Ksamples-epoch2031-best071.pt" #RETRAINED
    elif transf_id == "mediumdepth":
        return "../weights/baseline3/baseline3-depth-model-DAVE2v3-108x192-2085epoch-64batch-50Ksamples-epoch1706-best013.pt"
    elif transf_id == "resinc":
        # return "../weights/baseline3/model-baseline3-resinc-ONLY4523EPOCHS-DAVE2v3-270x480-5000epoch-64batch-50Ksamples-epoch4400-best149.pt"
        # return "../weights/baseline3/portal432639-baseline3-resinc-Tset-DAVE2v3-270x480-5000epoch-64batch-50Ksamples-epoch4844-best073.pt" # RETRAINED
        return "../weights/baseline3/resinc-test-model-DAVE2v3-270x480-5000epoch-64batch-10Ksamples-epoch001-best002.pt" # TEST
    elif transf_id == "resdec":
        # return "../weights/baseline3/baseline3-resdec-model-DAVE2v3-54x96-5000epoch-64batch-50Ksamples-epoch4612-best086.pt"
        return "../weights/baseline3-10K/portal644260_resdec_Tset-54x96-5000epoch-64batch-10Ksamples-epoch4826-best115.pt"


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
    df = pd.read_csv("./config-segments_inuse-revised.csv")
    hash = randstr()
    df = df.reset_index()  # make sure indexes pair with number of rows
    random.seed(1703)
    args = parse_args()
    vqvaes = [
              # "../weights/baseline3-10K/CHECKED-portal628217-depth-10K-DAVE2v3-108x192-75Ksamples-epoch1022-best032.pt",
              # "../weights/baseline3-10K/portal644260_resdec_Tset-54x96-5000epoch-64batch-10Ksamples-epoch4826-best115.pt",
              # "../weights/baseline3-50K/portal409517_baseline3_Tset50K_fisheye-DAVE2v3-108x192-5000epoch-64batch-50Ksamples-epoch2031-best071.pt",
              # "../weights/baseline3-50K/portal432639-baseline3-resinc-Tset-DAVE2v3-270x480-5000epoch-64batch-50Ksamples-epoch4844-best073.pt", # error on
              # 2023-09-24 20:32:43,460 ERROR    Uncaught exception:
              # Traceback (most recent call last):
              #   File "C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation/simulation/take-a-lap-test-baseline3-withconfig.py", line 477, in <module>
              #     results = main(config_topo_id, spawn_pos, rot_quat, cluster, model_name, hash=hash, transf_id=effects[i], cuton_pt=cuton_pt, cutoff_pt=cutoff_pt)
              #   File "C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation/simulation/take-a-lap-test-baseline3-withconfig.py", line 317, in main
              #     results = run_scenario(vehicle, bng, scenario, model, default_scenario=default_scenario, road_id=road_id, seg=seg,
              #   File "C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation/simulation/take-a-lap-test-baseline3-withconfig.py", line 237, in run_scenario
              #     prediction = model(processed_img.type(torch.float32))
              #   File "C:\Users\Meriel\Documents\GitHub\supervised-universal-transformation\venv-sutransf\lib\site-packages\torch\nn\modules\module.py", line 1051, in _call_impl
              #     return forward_call(*input, **kwargs)
              #   File "C:\Users\Meriel\Documents\GitHub\supervised-universal-transformation\simulation\DAVE2pytorch.py", line 253, in forward
              #     x = self.lin1(x)
              #   File "C:\Users\Meriel\Documents\GitHub\supervised-universal-transformation\venv-sutransf\lib\site-packages\torch\nn\modules\module.py", line 1051, in _call_impl
              #     return forward_call(*input, **kwargs)
              #   File "C:\Users\Meriel\Documents\GitHub\supervised-universal-transformation\venv-sutransf\lib\site-packages\torch\nn\modules\linear.py", line 96, in forward
              #     return F.linear(input, self.weight, self.bias)
              #   File "C:\Users\Meriel\Documents\GitHub\supervised-universal-transformation\venv-sutransf\lib\site-packages\torch\nn\functional.py", line 1847, in linear
              #     return torch._C._nn.linear(input, weight, bias)
              # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x384 and 128x500)

              # "../weights/baseline3-50K/portal522804-resdec-50K-DAVE2v3-54x96-epoch2509-best063.pt",

                #TODOS 9/26:
                "../weights/baseline3-75K/portal628179-depth-75K-DAVE2v3-108x192-5000epoch-epoch1764-best038.pt",
                # "../weights/baseline3-10K/CHECKED-portal628217-depth-10K-DAVE2v3-108x192-75Ksamples-epoch1022-best032.pt",
                # "../weights/baseline3-50K/portal442655-baseline3-depth-50K-Tset-DAVE2v3-108x192-5000epoch-64batch-epoch4572-best073.pt",
              ]
    effects = ["mediumdepth", "resdec", "fisheye", "resinc", "resdec"]
    for i in range(len(vqvaes)):
        for index, row in df.iterrows():
            config_topo_id = row["TOPOID"]
            spawn_pos = parse_list_from_string(row["SPAWN"])
            rot_quat = parse_list_from_string(row["ROT_QUAT"])
            cluster = row["SEGNUM"]
            cutoff = parse_list_from_string(row["END"])
            model_name = vqvaes[i] #get_model_name(args.effect)
            cuton_pt = parse_list_from_string(row["START"])
            cutoff_pt = parse_list_from_string(row["CUTOFF"])
            # results = main(config_topo_id, spawn_pos, rot_quat, cluster, model_name, hash=hash, transf_id=args.effect, cuton_pt=cuton_pt, cutoff_pt=cutoff_pt)
            results = main(config_topo_id, spawn_pos, rot_quat, cluster, model_name, hash=hash, transf_id=effects[i], cuton_pt=cuton_pt, cutoff_pt=cutoff_pt)