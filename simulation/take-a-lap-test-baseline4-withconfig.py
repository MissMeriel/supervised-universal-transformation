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
import torchvision.transforms as T

# globals
integral, prev_error = 0.0, 0.0
overall_throttle_setpoint = 40
setpoint = overall_throttle_setpoint
lanewidth = 3.75 #2.25
centerline = []
centerline_interpolated = []
roadleft = []
roadright = []

import argparse

def parse_args():
    parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does',
                                     epilog='Text at the bottom of help')
    parser.add_argument('--effect', help='image transformation', default=None)
    args = parser.parse_args()
    print(f"cmd line args:{args}")
    return args


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
    actual_middle, adjusted_middle = road_analysis(bng, road_id)
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
                 beamnginstance='C:/Users/Meriel/Documents/BeamNG.researchINSTANCE4', port=64956, topo_id=None):
    global base_filename, centerline_interpolated, centerline
    # random.seed(1703)
    setup_logging()
    beamng = BeamNGpy('localhost', port, home='F:/BeamNG.research.v1.7.0.1', user=beamnginstance)
    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model, licence='EGO', color=default_color)
    vehicle = setup_sensors(vehicle, img_dims, fov=fov)
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
    edges = bng.get_road_edges(road_id)
    centerline = [edge['middle'] for edge in edges]
    point_nearest = get_nearest_point(centerline, spawn_pos)
    print(f"{point_nearest=}")
    bng.pause()
    assert vehicle.skt
    return vehicle, bng, scenario


def run_scenario(vehicle, bng, scenario, model, default_scenario, road_id, reverse=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), seg=None, vqvae=None,
                 transf=None, detransf=None, topo=None, cuton_pt=None, cutoff_pt=None):
    global base_filename, centerline_interpolated
    global integral, prev_error, setpoint
    orig_model_image_size = (108, 192)
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
    # image.save(f"./start-{topo}-{transf}.jpg")
    print(f"{transf=} \t {detransf=}")
    damage = wheelspeed = kph = throttle = runtime = distance_from_center = 0.0
    prev_error = setpoint
    kphs = []; traj = []; steering_inputs = []; throttle_inputs = []; timestamps = []
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
        if "depth" in transf:
            image = transformations.blur_with_depth_image(np.array(image), np.array(image_depth))
        try:
            processed_img = model.process_image(image).to(device, dtype=torch.float)
        except:
            processed_img = transform(np.asarray(image))[None]
        if vqvae is not None:
            processed_img = processed_img.to(device, dtype=torch.float)
            embedding_loss, processed_img, perplexity = vqvae(processed_img)
        if transf == "resinc" or transf == "resdec":
            # print(f"{processed_img.shape=}")
            processed_img = T.Resize(size=orig_model_image_size)(processed_img)
            # print(f"{processed_img.shape=}")

        prediction = model(processed_img)
        steering = float(prediction.item())
        vqvae_viz = processed_img.cpu().detach()[0]
        vqvae_viz = vqvae_viz.permute(1,2,0)
        cv2.imshow('car view', vqvae_viz.numpy()[:, :, ::-1])
        cv2.waitKey(1)
        total_imgs += 1
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        dt = (sensors['timer']['time'] - start_time) - runtime
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
        # print(f"{outside_track=} \t{distance_from_center=:1f}")
        if outside_track:
            print("Left track, exiting...")
            break

    cv2.destroyAllWindows()
    track_len = get_distance_traveled(centerline)
    deviation = calc_deviation_from_center(centerline, traj)
    results = {'runtime': round(runtime,3), 'damage': damage, 'kphs':kphs, 'traj':traj, 'final_img':final_img, 'deviation':deviation,
               "track_length":track_len
               }
    return results


def zero_globals():
    global centerline, centerline_interpolated, roadleft, roadright
    centerline = []
    centerline_interpolated = []
    roadleft = []
    roadright = []


from vqvae.models.vqvae import VQVAE
import torchsummary
def main(topo_id, spawn_pos, rot_quat, vqvae_name, count, cluster="000", hash="000", transf_id=None, detransf_id=None, cuton_pt=None, cutoff_pt=None, arch_id=None):
    global base_filename, centerline, centerline_interpolated
    zero_globals()
    model_name="../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_name, map_location=device).eval()

    vqvae = VQVAE(128, 32, 2, 512, 64, .25, transf=transf_id, arch_id=arch_id).eval().to(device)
    checkpoint = torch.load(vqvae_name, map_location=device)
    vqvae.load_state_dict(checkpoint["model"])
    # torchsummary.summary(vqvae, (3, 108, 192))
    vqvae_id = f"baseline4{count}K"
    default_scenario, road_id, seg, reverse = get_topo(topo_id)
    img_dims, fov, transf = get_transf(transf_id)
    runs = 25
    vqvae_literal = vqvae_name.split("/")[-1].replace('.pth', '')
    filepathroot = f"simresults/bestfisharchreruns-{runs}RUNS-{vqvae_id}-{transf_id}-arch{arch_id}-{hash}/{vqvae_literal}/{topo_id}topo-cluster{cluster}-{runs}runs-{hash}/"
    print(f"{filepathroot=}")
    Path(filepathroot).mkdir(exist_ok=True, parents=True)

    print(f"TOPO_ID={topo_id} \tTRANSFORM={transf_id} \t IMAGE DIMS={img_dims}")
    random.seed(1703)
    vehicle, bng, scenario = setup_beamng(default_scenario, spawn_pos, rot_quat, road_id=road_id, seg=seg, reverse=reverse, img_dims=img_dims, fov=fov, vehicle_model='hopper',
                                          beamnginstance='F:/BeamNG.researchINSTANCE3', port=64356, topo_id=topo_id)
    if topo_id == "extra_windingtrack" or topo_id == "Rturncommercialunderpass":
        centerline.reverse()
        centerline_interpolated.reverse()
    distances, deviations, deviations_seq, trajectories, runtimes = [], [], [], [], []

    for i in range(runs):
        results = run_scenario(vehicle, bng, scenario, model, default_scenario=default_scenario, road_id=road_id, seg=seg,
                               vqvae=vqvae, transf=transf_id, detransf=detransf_id, topo=topo_id, cuton_pt=cuton_pt, cutoff_pt=cutoff_pt)
        results['distance'] = get_distance_traveled(results['traj'])
        # plot_trajectory(results['traj'], f"{default_scenario}-{model._get_name()}-{road_id}-runtime{results['runtime']:.2f}-dist{results['distance']:.2f}")
        print(f"\nBASE MODEL {transf_id}  {topo_id} RUN {i}:"
              f"\n\tdistance={results['distance']:1f}"
              f"\n\tavg dist from center={results['deviation']['mean']:3f}")
        distances.append(results['distance'])
        deviations.append(results['deviation']['mean'])
        deviations_seq.append(results['deviation'])
        trajectories.append(results["traj"])
        runtimes.append(results['runtime'])
    summary = {
        "trajectories": trajectories,
        "dists_from_centerline": deviations,
        "dists_travelled": distances,
        "track_length": results["track_length"],
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
        "vqvae_id": vqvae_id,
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
    id = f"basemodelalone-{vqvae_id}"
    try:
        plot_deviation(trajectories, centerline, roadleft, roadright, "DAVE2V3 ", filepathroot, savefile=f"{topo_id}-{transf_id}-{id}")
    except:
        plot_deviation(trajectories, centerline, roadleft, roadright, "DAVE2V3", filepathroot, savefile=f"{topo_id}-{transf_id}-{id}")
    bng.close()
    del vehicle, bng, scenario
    return summary


def summarize_results(all_results):
    distances, deviations, avg_percentage, avgd_distances = [], [], [], []
    trackcount = 5
    for i, result in enumerate(all_results):
        distances.extend(result["dists_travelled"])
        deviations.extend(result["dists_from_centerline"])
        avgd_distances.append(sum(distances[i]) / len(distances[i]))
        x = [d / result["track_length"] for d in distances[i]]
        avg_percentage += sum(x) / trackcount
    print(f"5-TRACK SUMMARY:"
          f"\n\tAvg. distance: {(sum(distances)/len(distances)):.1f}"
          f"\n\tAvg. distance deviation: {np.std(distances):.1f}"
          f"\n\tAvg. center deviation: {(sum(deviations) / len(deviations)):.3f}"
          f"\n\tAvg. track %% travelled: {((avg_percentage / trackcount) * 100):.1f}%"
          f"\n\tTotal distance travelled: {sum(avgd_distances):.1f}"
    )


def get_vqvae_name(transf_id, count=10):
    if transf_id == "mediumfisheye" and count == 10:
        # return "../weights/baseline4-10K/vqvae_UUST_fisheye_samples10000_epochs500_52054974_tue_aug_8_03_54_40_2023.pth"
        return "../weights/baseline4-10K/vqvae_RQ1v2_fisheye_samples10000_epochs500_5444459_sat_jul_22_03_21_07_2023.pth"
    elif transf_id == "mediumdepth" and count == 10:
        return "../weights/baseline4-10K/vqvae_UUST_depth_samples10000_epochs500_52054974_mon_aug_7_10_16_44_2023.pth"
    elif transf_id == "resinc" and count == 10:
        # return "../weights/baseline4-10K/vqvae_UUST_resinc_samples10000_epochs309_52614507_mon_aug_21_15_14_38_2023.pth"
        return "../weights/baseline4-10K/portal409110_vqvae10K_predlossweight1.0_resinc_bestmodel486.pth"
    elif transf_id == "resdec" and count == 10:
        # return "../weights/baseline4-10K/vqvae_UUST_resdec_samples10000_epochs500_52036393_mon_aug_7_00_37_04_2023.pth"
        return "../weights/baseline4-10K/portal453967_vqvae10K_resdec_newarch_predlossweight1.0_bestmodel479.pth"

    elif transf_id == "resdec" and count == 50:
        return "../weights/baseline4-50K/portal453968_vqvae50K_resdec_newarch_predloss1.0_bestmodel415.pth"

    elif transf_id == "mediumfisheye" and count == 95:
        return "../weights/baseline4-95K/vqvae95K_fisheye_epoch178.pth"
    elif transf_id == "mediumdepth" and count == 95:
        return "../weights/baseline4-95K/vqvae95K_depth_epoch258.pth"
    elif transf_id == "resinc" and count == 95:
        return "../weights/baseline4-95K/vqvae95K-ONLY279EPOCHS_resinc_epoch86.pth"
    elif transf_id == "resdec" and count == 95:
        return "../weights/baseline4-95K/vqvae95K_resdec_epoch499.pth"


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    df = pd.read_csv("./config-segments_inuse-revised.csv")
    # detransf_ids = ["mediumdepth", "mediumfisheye","resinc", "resdec", ]
    # hashes = [randstr() for t in detransf_ids]
    hash = randstr()
    count = 25
    arch_id = None

    # EXPERIMENTAL VQVAE ENCODER ARCHS
    vqvae_name = "../weights/baseline4-10K/portal743150-vqvae_fisheye_newencoderarch2_samples10000_pred_weight1.0_bestmodel483.pth" # SIMMED, BEST SO FAR
    # vqvae_name = "../weights/baseline4-50K/portal752565_vqvae50K_fisheye_newencoderarch2_predweight1.0_bestmodel493.pth" # SIMMED, BEST SO FAR
    # vqvae_name = "../weights/baseline4-10K/portal743149_vqvae_UUST_fisheye_newencoderarch1_samples10000_pred_weight1.0_epochs500_bestmodel498.pth" # SIMMED
    # vqvae_name = "../weights/baseline4-10K/portal743151_vqvae_fisheye_newencoderarch3_samples10000_pred_weight1.0_epochs500_bestmodel493.pth" # SIMMED
    # vqvae_name = "../weights/baseline4-10K/portal743152_vqvae_depth_newencoderarch2_samples10000_predweight1.0_epochs500_bestmodel482.pth" # SIMMED
    # vqvae_name = "../weights/baseline4-10K/portal743153_vqvae_depth_newencoderarch1_samples10000_predweight1.0_epochs500_bestmodel498.pth" # SIMMED, BEST SO FAR
    # vqvae_name = "../weights/baseline4-50K/portal752574_vqvae_depth_newencoderarch1_samples50K_predweight1.0_bestmodel460.pth" # SIMMED, BEST SO FAR
    # vqvae_name = "../weights/baseline4-50K/portal752572_vqvae50K_UUST_fisheye_newencoderarch3_predweight1.0_bestmodel436.pth" # SIMMED


    # SECOND ROUND OF EXPERIMENTAL ENCODER ARCHS
    # vqvae_name = "../weights/baseline4-10K/portal864823_vqvae10K_fisheye_newencoderarch4_predweight1.0_bestmodel499.pth"
    # vqvae_name = "../weights/baseline4-10K/portal864822_vqvae10K_depth_newencoderarch4_predweight1.0_bestmodel450.pth"
    # vqvae_name = "../weights/baseline4-10K/portal863194_vqvae10K_depth_newencoderarch3_predweight1.0_bestmodel490.pth" # 0.014483165 lower training loss, use this one
    # vqvae_name = "../weights/baseline4-10K/portal863175_vqvae10K_depth_newencoderarch3_predweight1.0_bestmodel499.pth" # 0.014921573 training loss

    # THIRD ROUND
    vqvae_names = [#"../weights/baseline4-10K/portal743150_vqvae_fisheye_newencoderarch2_samples10K_pred_weight1.0_epochs500_bestmodel483.pth",
                   # "../weights/baseline4-10K/portal743149_vqvae_UUST_fisheye_newencoderarch1_samples10000_pred_weight1.0_epochs500_bestmodel498.pth"
                   "../weights/baseline4-50K/portal752565_vqvae50K_fisheye_newencoderarch2_predweight1.0_bestmodel493.pth"
                    ]
    df = df.reset_index()  # make sure indexes pair with number of rows

    # for i, detransf_id in enumerate(detransf_ids):
    random.seed(1703)
    args = parse_args()
    # detransf_id = args.effect
    effects = ["mediumfisheye", "mediumfisheye"]
    arch_ids = [2, 1]
    for i, vqvae_name in enumerate(vqvae_names):
        detransf_id = effects[i]
        arch_id = arch_ids[i]
        for index, row in df.iterrows():
            # vqvae_name = get_vqvae_name(detransf_id, count=count)
            print(f"{vqvae_name=}")
            config_topo_id = row["TOPOID"]
            spawn_pos = parse_list_from_string(row["SPAWN"])
            rot_quat = parse_list_from_string(row["ROT_QUAT"])
            cluster = row["SEGNUM"]
            cutoff = parse_list_from_string(row["END"])
            cuton_pt = parse_list_from_string(row["START"])
            cutoff_pt = parse_list_from_string(row["CUTOFF"])
            results = main(config_topo_id, spawn_pos, rot_quat, vqvae_name, count, cluster=cluster, hash=hash, transf_id=detransf_id, cuton_pt=cuton_pt, cutoff_pt=cutoff_pt, arch_id=arch_id)