import sys

sys.path.append("C:/Users/Meriel/Documents/GitHub/BeamNGpy/src")
sys.path.append("C:/Users/Meriel/Documents/GitHub/IFAN")

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


# from ast import literal_eval
# from wand.image import Image as WandImage
import DAVE2pytorch
from DAVE2pytorch import DAVE2PytorchModel, DAVE2v3
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
from sim_utils import *
from transformations import transformations
from transformations import detransformations

# C:\Users\Meriel\Documents\GitHub\supervised-universal-transformation\venv-sutransf\Scripts\python.exe C:\Users\Meriel\Documents\GitHub\supervised-universal-transformation\simulation\test-validation-tracks-baseline000.py

# globals
integral, prev_error = 0.0, 0.0
overall_throttle_setpoint = 40
setpoint = overall_throttle_setpoint
lanewidth = 3.75 #2.25
centerline = []
centerline_interpolated = []
roadleft = []
roadright = []


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
    print("spawn point:{}".format(spawn))
    print("beginning of script:{}".format(middle[0]))
    # plot_trajectory(traj, "Points on Script (Final)", "AI debug line")
    # centerline = copy.deepcopy(traj)
    centerline_interpolated = copy.deepcopy(traj)
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return line, bng


def setup_beamng(default_scenario, road_id, reverse=False, seg=1, img_dims=(240,135), fov=51, vehicle_model='etk800', default_color="green", steps_per_sec=15,
                 beamnginstance='C:/Users/Meriel/Documents/BeamNG.researchINSTANCE4', port=64956):
    global base_filename

    random.seed(1703)
    setup_logging()
    print(road_id)
    beamng = BeamNGpy('localhost', port, home='C:/Users/Meriel/Documents/BeamNG.research.v1.7.0.1', user=beamnginstance)
    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model, licence='EGO', color=default_color)
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
    kphs, traj, steering_inputs, timestamps, throttle_inputs = [], [], [], [], []
    damage = 0.0
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
                image = transformations.blur_with_depth_image(np.array(image), np.array(image_depth))
            elif "res" in transf:
                image = image.resize((192, 108))

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

        dists = dist_from_line(centerline, vehicle.state['pos'])
        m = np.where(dists == min(dists))[0][0]
        if damage > 1.0:
            print(f"Damage={damage:.3f}, exiting...")
            break
        bng.step(1, wait=True)

        if distance2D(vehicle.state["pos"], cutoff_point) < 12 and len(traj) > 30:
            print("Reached cutoff point, exiting...")
            break

        outside_track, distance_from_center = has_car_left_track(vehicle.state['pos'], centerline_interpolated, max_dist=6.0)
        if outside_track:
            print("Left track, exiting...")
            break

    cv2.destroyAllWindows()

    deviation = calc_deviation_from_center(centerline, traj)
    results = {'runtime': round(runtime,3), 'damage': damage, 'kphs':kphs, 'traj':traj,
               'deviation':deviation, "steering_inputs":steering_inputs,
                "throttle_inputs":throttle_inputs, "timestamps":timestamps,
               }
    return results


def zero_globals():
    global centerline, centerline_interpolated, roadleft, roadright
    centerline = []
    centerline_interpolated = []
    roadleft = []
    roadright = []


def main(topo_id, hash="000", detransf_id=None, transf_id=None):
    global base_filename
    zero_globals()
    random.seed("1703")
    model_name = "../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_name, map_location=device).eval()
    vqvae_name = None
    vqvae = None
    vqvae_id = "baseline000"
    default_scenario, road_id, seg, reverse = get_topo(topo_id)
    img_dims, fov, transf = get_transf(transf_id)
    print(f"TRANSFORM={transf_id} \t IMAGE DIMS={img_dims}")

    vehicle, bng, scenario = setup_beamng(default_scenario=default_scenario, road_id=road_id, seg=seg, reverse=reverse, img_dims=img_dims, fov=fov, vehicle_model='hopper',
                                          beamnginstance='F:/BeamNG.researchINSTANCE3', port=64956)
    distances, deviations, trajectories, runtimes = [], [], [], []
    steerings, throttles, timestamps = [], [], []
    runs = 3

    filepathroot = f"F:/safetransf-base-model-validation-tracks/{model_name.split('/')[-1].replace('.pt', '')}-{hash}/{topo_id}/{topo_id}-{transf_id}-{default_scenario}-{road_id}-{runs}runs-{hash}/"
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
        runtimes.append(results['runtime'])
        steerings.append(results["steering_inputs"])
        throttles.append(results["throttle_inputs"])
        timestamps.append(results["timestamps"])
    summary = {
        "trajectories": trajectories,
        "dists_from_centerline": deviations,
        "dists_travelled": distances,
        "steering_inputs": steerings,
        "throttle_inputs": throttles,
        "timestamps": timestamps,
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
        "runtimes": runtimes
    }

    picklefile = open(f"{filepathroot}/summary-{model_name.split('/')[-1]}_{vqvae_id}.pickle", 'wb')
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
    id = f"DAVE2V3-{vqvae_id}"
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

def load_tracklist(filename="posefiles/validation-tracks.txt"):
    with open(filename, "r") as f:
        tracks = f.readlines()
        tracks = [t.replace("\n", "") for t in tracks]
    return tracks

if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    transf_id = "medium"
    all_results = []
    # DONE:
    tracks = ["extra_small_islandcoast_a_nw","extra_jungleouter_road_a", "extra_jungledrift_road_d", "extra_westcoastrocks", "extra_jungleouter_road_b",
                "Lturn_uphill", "extra_dock","countryrd", "Rturn_mtnrd",  "Rturn", "Lturn", "extra_utahlong",
                "extra_utahlong2", "extra_utahexittunnel", "extra_utahswitchback", "Rturn_small_island_ai_1", "Rturn_int_a_small_island",
                "extra_junglemountain_road_c",  "Rturn_industrialnarrowservicerd", "Rturnrockylinedmtnroad",
                "extra_dock", "Rturn_maintenancerd", "Rturn_narrowcutthru", "Rturn_bigshoulder", "Rturn_servicecutthru",
                "extrawinding_industrialrcasphalta", "extrawinding_industrial7978","Rturn_hirochitrack", "Rturn_sidequest", "Rturn_lanelines",
                "Rturn_bridge", "Lturn_narrowservice", "Rturn_industrialrc_asphaltd", "Rturn_industrial7978",
                "Rturn_industrialrc_asphaltb", "Lturn_junglemountain_road_e", "extra_jungledrift_road_b", "extra_jungle8161",
                "extra_junglemountain_alt_f", "extra_junglemountain_road_i", "extra_junglemeander8114", "extra_jungledrift_road_m",
                "extra_jungledrift_road_k", "extra_jungle8131", "extra_junglemountain_alt_a", "extra_junglemeander7994", "extra_jungle8000",
                "extra_dock", "extra_winding",  "extra_whatever", "extra_utahtunnel", "extra_wideclosedtrack",
                "extra_wideclosedtrack2", "extra_windingnarrowtrack", "extra_windingtrack", "extra_multilanehighway", "extra_multilanehighway2",
                "extra_jungleouter_road_c", "extrawinding_industrialtrack", "straight",
                "Lturn_test3", "extra_driver_trainingvalidation2", "extra_lefthandperimeter",
                "narrowjungleroad1", "narrowjungleroad2", "Lturnyellow", "straightcommercialroad", "Rturninnertrack", "straightwidehighway",
                "Rturncommercialunderpass", "Lturncommercialcomplex", "Lturnpasswarehouse", "Rturnserviceroad", "Rturnlinedmtnroad",
                "Rturn_industrial8022whitepave", "Rturn_industrial8068widewhitepave", "Rturn_industrialrc_asphaltc",
                "extra_westdockleftside", "extra_westmtnroad", "extra_jungledrift_road_f", "extra_junglemain_tunnel", "extra_jungledrift_road_s",
                "extra_jungledrift_road_e", "extra_jungledrift_road_a", "extra_junglemountain_road_h", "extra_westoutskirts", "extra_westsuburbs",
                "extra_westunderpasses", "extra_westLturnway", "extra_westofframp",
                "dealwithlater"]
    # FIX: , "extrawinding_industrial8067","extra_windyjungle8082", "extra_utahswitchback"
    # FIX 4 VALIDATION: "countryroad", "Rturn_narrowcutthru"
    #topos = ["extra_test7", "Lturn_uphill", "extra_test2", "extra_test1",  "extra_test3", "extra_test1", "extra_test4",]
    tracks = load_tracklist("posefiles/validation-tracks2.txt")
    for track in tracks:
        hash = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        results = main(track, hash=hash, transf_id=transf_id)
        all_results.append(results)
        # results = main("extra_windingnarrowtrack", hash=hash, transf_id=transf_id)
        # all_results.append(results)
        # results = main("extra_windingtrack", hash=hash, transf_id=transf_id)
        # all_results.append(results)
        # results = main("Rturn_bigshouldertopo", hash=hash, transf_id=transf_id)
        # all_results.append(results)
        # results = main("Rturn_bridgetopo", hash=hash, transf_id=transf_id)
        # all_results.append(results)
    summarize_results(all_results)