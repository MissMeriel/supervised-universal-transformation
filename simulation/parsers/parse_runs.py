import numpy as np
from matplotlib import pyplot as plt
import shutil
import logging, random, string, time, os
# from tabulate import tabulate
from pathlib import Path
import argparse, sys

import statistics, math
from ast import literal_eval

import torch
from PIL import Image
import kornia
import pickle
import warnings
from functools import wraps

# clustering algos
import pandas as pd
import numpy as np
import random
# from kneed import KneeLocator
# from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner


def randstr():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", '--dataset', type=str, default="/p/sdbb/safetransf-base-model-validation-tracks/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051-7ZYMOA", help='parent directory of results dataset')
    # parser.add_argument('-f', '--figure', type=str, help='table or figure to generate (choose from: table1 table2 table3 table5 table6 figure5 figure6 figure7 figure8)')
    # parser.add_argument('-r', '--threads', type=int, default=100, help='num. threads')
    args = parser.parse_args()
    return args


def unpickle_results(filename):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results


@ignore_warnings
def lineseg_dists(p, a, b):
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)
    h = np.maximum.reduce([s, t, np.zeros(len(s))])
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
    return np.hypot(h, c)


def dist_from_line(centerline, point):
    a = [[x[0], x[1]] for x in centerline[:-1]]
    b = [[x[0], x[1]] for x in centerline[1:]]
    a = np.array(a)
    b = np.array(b)
    dist = lineseg_dists([point[0], point[1]], a, b)
    return dist


def plot_deviation(trajectory, centerline, left, right, xlim=None, ylim=None, resultsdir="images", topo_id=None, title=None):
    fig, ax = plt.subplots()
    x = [point[0] for point in trajectory]
    y = [point[1] for point in trajectory]
    plt.plot(x, y, label="Traj", linewidth=5)
    x = [point[0] for point in centerline]
    y = [point[1] for point in centerline]
    plt.plot(x, y, 'k')
    x = [point[0] for point in left]
    y = [point[1] for point in left]
    plt.plot(x, y, 'k')
    x = [point[0] for point in right]
    y = [point[1] for point in right]
    plt.plot(x, y, 'k')
    if title is None:
        plt.title(f'Trajectories for {topo_id}', fontdict={'fontsize': 10})
    else:
        plt.title(f'Trajectories for {title}', fontdict={'fontsize': 10})
    # plt.legend()
    if xlim is not None and ylim is not None:
        plt.xlim(xlim)
        plt.ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("{}/{}-{}.jpg".format(resultsdir, topo_id, randstr()))
    plt.close("all")
    del x, y


def get_nearest_point(line, point):
    dists = dist_from_line(line, point)
    # print(len(dists))
    # print(dists[0:10])
    # index = line.index(minimum_dist)
    index = np.nanargmin(np.array(dists))
    # print("Nearest Index position: ",index)
    line_point = line[index]
    return line_point, index


def distance2D(a,b):
    return math.sqrt(math.pow(a[0]- b[0], 2) + math.pow(a[1] - b[1], 2))


def get_traj_length(traj):
    length = sum([distance2D(a,traj[i+1]) for i,a in enumerate(traj[:-1])])
    return length


# find a trajectory window with high deviation on 100m of track
def calc_track_and_traj_window(cluster_center, centerline, trajectory, delta=120):
    delta_start = delta_end = delta
    centerline_delta = 2.5 # 2.5 10
    traj_window = [np.max([0, cluster_center - delta_start]), np.min([len(trajectory)-1, cluster_center + delta_end])]
    point_start, index_start = get_nearest_point(centerline, trajectory[traj_window[0]])
    point_end, index_end = get_nearest_point(centerline, trajectory[traj_window[1]])
    centerline_length = get_traj_length(centerline[index_start:index_end])
    count = 0
    reversed = False
    while centerline_length <= 100 or centerline_length > 100 + centerline_delta:
        # if the length of track is too short
        if centerline_length - 100 > centerline_delta:
            if index_start == 0:
                index_end -= 1
            elif index_end == len(centerline)-10:
                index_start += 1
            else:
                index_start += 1
        # if the length of track is too long
        elif centerline_length - 100 < 0:
            print(f"\t{index_start=} {index_end=} {len(centerline)=} {centerline_length=}")
            if index_start == 0:
                index_end += 1
            elif index_end == len(centerline)-10:
                index_start -= 1
            else:
                index_start -= 1
        if index_start > index_end:
            reversed = True
            centerline.reverse()
            trajectory.reverse()
            temp=index_start
            index_start = index_end
            index_end = temp
        centerline_length = get_traj_length(centerline[index_start:index_end])
        print(f"2. {centerline_length=}")
        if count > 250:
            print(f"Maxed out window adjustment iterations, exiting.")
            # exit(0)
            return None, None
        count += 1

    return [index_start, index_end], reversed


# cluster center by trajectory index
def cluster_to_road_def(results, cluster_center):
    track_window, reversed =calc_track_and_traj_window(cluster_center, results['centerline_interpolated'], results['trajectories'][0], delta=120)
    if track_window is None:
        return None
    center_index_start = np.max([0, track_window[0]]) 
    center_index_end = np.min([len(results['centerline_interpolated'])-1, track_window[1]])
    centerline = results['centerline_interpolated']
    roadleft = results['roadleft']
    roadright = results['roadright']
    trajectory = results['trajectories'][0]
    if reversed:
        centerline.reverse()
        roadleft.reverse()
        roadright.reverse()
    roadside_point_start, roadside_index_start = get_nearest_point(roadleft, centerline[track_window[0]])
    roadside_point_end, roadside_index_end = get_nearest_point(roadleft, centerline[track_window[1]])
    roadside_index_start = np.max([0, roadside_index_start-1]) 
    roadside_index_end = np.min([len(roadleft)-1, roadside_index_end+2])
    
    traj_point_start, traj_index_start = get_nearest_point(results['trajectories'][0], centerline[center_index_start] )
    traj_point_end, traj_index_end = get_nearest_point(results['trajectories'][0], centerline[center_index_end] )
    road_def = {
                "roadleft": roadleft[roadside_index_start:roadside_index_end], 
                "roadright": roadright[roadside_index_start:roadside_index_end], 
                "centerline":centerline[center_index_start:center_index_end], 
                "trajectory": results['trajectories'][0][traj_index_start:traj_index_end]
                }
    return road_def


def get_deviation(trajectory, centerline):
    dists = []
    for point in trajectory:
        try:
            dist = dist_from_line(centerline, point)
            dists.append(min(dist))
        except:
            print(f"centerline shape={np.array(centerline).shape}")
            print(f"point shape={np.array(point).shape}")
            exit(0)
    # return np.std(dists)
    return np.mean(dists)

@ignore_warnings
def write_results(training_file, results):
    with open(training_file, "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

# trajectories <class 'list'>
# dists_from_centerline <class 'list'>
# dists_travelled <class 'list'>
# steering_inputs <class 'list'>
# throttle_inputs <class 'list'>
# timestamps <class 'list'>
# img_dims <class 'tuple'>
# centerline_interpolated <class 'list'>
# roadleft <class 'list'>
# roadright <class 'list'>
# default_scenario <class 'str'>
# road_id <class 'str'>
# topo_id <class 'str'>
# transf_id <class 'str'>
# vqvae_name <class 'NoneType'>
# model_name <class 'str'>
# runtimes <class 'list'>
def main(args):
    allDirs = [f"{args.dataset}/{_}" for _ in os.listdir(f"{args.dataset}") if os.path.isdir(f"{args.dataset}/{_}")] # and "extra_driver_trainingvalidation2" in _]
    fileExt = f".pickle"
    n_clusters = 10
    newdir = f"./high-dev-roadsegs-{n_clusters:02}clusters-{randstr()}/"
    os.makedirs(newdir, exist_ok=True)
    for dir in allDirs:
        subDirs = [f"{dir}/{_}" for _ in os.listdir(f"{dir}") if os.path.isdir(f"{dir}/{_}")]
        for subdir in subDirs:
            results_files = [_ for _ in os.listdir(subdir) if _.endswith(fileExt)]
            for rf in results_files:
                results = unpickle_results("/".join([subdir, rf]))
                print(f"\n\n{results['topo_id']}")# {len(results['roadleft'])}")
                print("/".join([subdir, rf]))
                topo_dir = f"{newdir}/{results['topo_id']}"
                os.makedirs(topo_dir)
                print("avg dist from centerline run 0", np.mean(results["dists_from_centerline"]))
                print("max dist from centerline run 0", np.max(results["dists_from_centerline"]))
                print("avg dists from centerline",np.sort(results['dists_from_centerline']))
                dists = []
                # end 10 before the end to avoid part where it runs off the road
                for point in results["trajectories"][0][:-10]:
                    dist = dist_from_line(results["centerline_interpolated"], point)
                    dists.append(round(min(dist), 1))
                print(f"{results['dists_travelled']=}")
                # find points with max deviation
                top_ct = len(dists) // 2 + 1
                top_devs = np.sort(dists)[-top_ct:]
                top_devs_idxs = np.argsort(dists)[-top_ct:]
                # pick points far away from each other
                # features = [[dev, idx] for dev, idx in zip(top_devs, top_devs_idxs)]
                features = [[dev, idx] for dev, idx in zip(dists, np.argsort(dists))]
                kmeans = KMeans(
                                init="random",
                                n_clusters=n_clusters,
                                n_init=100,
                                max_iter=300,
                                random_state=42
                )
                kmeans.fit(features)
                # print(f"{kmeans.cluster_centers_=}")

                # expand window to Xm stretch of road
                for i, center in enumerate(kmeans.cluster_centers_[:,1]):
                    print(f"\nFinding road def for cluster {i} ({center=})")
                    road_def = cluster_to_road_def(results, int(round(center, 0)))
                    if road_def is None:
                        print("Delta too small, continuing with next track")
                        continue
                    deviation = get_deviation(road_def["trajectory"], results["centerline_interpolated"])
                    traj_len = sum([distance2D(a,road_def["trajectory"][i+1]) for i,a in enumerate(road_def["trajectory"][:-1])])
                    ctrline_length=sum([distance2D(a,road_def["centerline"][i+1]) for i,a in enumerate(road_def["centerline"][:-1])])
                    plot_deviation(
                                    road_def["trajectory"], road_def["centerline"], road_def["roadleft"], road_def["roadright"], 
                                    xlim=None, ylim=None, 
                                    resultsdir=topo_dir, 
                                    topo_id=results['topo_id'] + "_cluster" + str(i), 
                                    title=f"{results['topo_id']} cluster{i} \n{ctrline_length=:.1f} traj_length={traj_len:.1f} {deviation=:.3f}"
                    )
                    road_def["deviation"] = deviation
                    road_def["traj_len"] = traj_len
                    road_def["ctrline_length"] = ctrline_length
                    print(f"{deviation=:.3f}")
                    print(f"{traj_len=:.1f}")
                    write_results(f"{topo_dir}/{results['topo_id']}-cluster{i}.pickle", road_def)
                # exit(0)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)



# TODO:
# fix extra_jungledrift_road_d centerline collapsing to len 0