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
    parser.add_argument("-t", '--dataset', type=str, default="/p/sdbb/safetransf-validation-runs/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051-7ZYMOA", help='parent directory of results dataset')
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


# cluster center by trajectory index
def cluster_to_road_def(results, cluster_center):
    delta = 120
    traj_window = [np.max([0, cluster_center-delta]), np.min([len(results['trajectories'][0])-1, cluster_center+delta])]
    # find road centerline nearest to beginning and end of traj window
    print(f"Finding centerline point nearest to traj point {results['trajectories'][0][traj_window[0]]}")
    point_start, index_start = get_nearest_point(results['centerline_interpolated'], results['trajectories'][0][traj_window[0]])
    print(f"Finding centerline point nearest to traj point {results['trajectories'][0][traj_window[1]]}")
    point_end, index_end = get_nearest_point(results['centerline_interpolated'], results['trajectories'][0][traj_window[1]])
    centerline = results['centerline_interpolated'][index_start:index_end] 
    roadleft = results['roadleft']
    roadright = results['roadright']
    roadside_point_start, roadside_index_start = get_nearest_point(roadleft, point_start)
    roadside_point_end, roadside_index_end = get_nearest_point(roadleft, point_end)
    roadside_index_start = np.max([0, roadside_index_start-1]) 
    roadside_index_end = np.min([len(roadleft)-1, roadside_index_end+2])
    if index_start > index_end:
        centerline = results['centerline_interpolated']
        centerline.reverse()
        point_start, index_start = get_nearest_point(centerline, results['trajectories'][0][traj_window[0]])
        point_end, index_end = get_nearest_point(centerline, results['trajectories'][0][traj_window[1]])
        centerline = centerline[index_start:index_end] 
        # print(f"{index_start=} {index_end=}")
        roadleft = results['roadleft']
        roadleft.reverse()
        roadright = results['roadright']
        roadright.reverse()
        roadside_point_start, roadside_index_start = get_nearest_point(roadleft, point_start)
        roadside_point_end, roadside_index_end = get_nearest_point(roadleft, point_end)
        roadside_index_start = np.max([0, roadside_index_start-1]) 
        roadside_index_end = np.min([len(roadleft)-1, roadside_index_end+2])
    print(f"{index_start=} {index_end=}")
    # roadside_point_start, roadside_index_start = get_nearest_point(results['roadleft'], point_start)
    # roadside_point_end, roadside_index_end = get_nearest_point(results['roadleft'], point_end)
    # roadside_index_start = np.max([0, roadside_index_start-1]) 
    # roadside_index_end = np.min([len(results['roadleft'])-1, roadside_index_end+2])
    road_def = {
                "roadleft": roadleft[roadside_index_start:roadside_index_end], 
                "roadright": roadright[roadside_index_start:roadside_index_end], 
                "centerline":centerline, 
                "trajectory": results['trajectories'][0][traj_window[0]:traj_window[1]]
                }
    return road_def


def get_deviation(trajectory, centerline):
    dists = []
    for point in trajectory:
        try:
            dist = dist_from_line(centerline, point)
        except:
            print(f"centerline shape={np.array(centerline).shape}")
            print(f"point shape={np.array(point).shape}")
            exit(0)
        dists.append(min(dist))
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
                tracklen = sum([distance2D(a,results["centerline_interpolated"][i+1]) for i,a in enumerate(results["centerline_interpolated"][:-1])])
                print(f"Track len.: \t\t{tracklen:.1f}")
                print(f"Avg. distance: \t\t{np.mean(results['dists_travelled']):.1f}")
                print(f"Avg. distance deviation:{np.std(results['dists_travelled']):.1f}")
                print(f"Avg. center deviation: \t{np.mean(results['dists_from_centerline']):.3f}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)



# TODO:
# fix extra_jungledrift_road_d centerline collapsing to len 0