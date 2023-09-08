import os
from pathlib import Path
import argparse
import pickle
import warnings
from functools import wraps
import numpy as np
from matplotlib import pyplot as plt
import shutil
import logging, random, string, time, os
# from tabulate import tabulate
from pathlib import Path
import argparse, sys
import statistics, math
from scipy.spatial.transform import Rotation as R

'''
Script to parse the simulation performance of baselines
on 100m segments of track
'''

warnings.filterwarnings("ignore")


def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner


CLUSTER_ROOT="/p/sdbb/safetransf-validation-runs/high-dev-roadsegs/high-dev-roadsegs-10clusters-delta2.5-10NIXS/"
cluster_paths = {
            "Lturnpasswarehouse":["Lturnpasswarehouse/Lturnpasswarehouse-cluster8.pickle",
                                        "Lturnpasswarehouse/Lturnpasswarehouse-cluster7.pickle",                                
                    ],
            "extra_driver_trainingvalidation2":["extra_driver_trainingvalidation2/extra_driver_trainingvalidation2-cluster1.pickle",
                                                "extra_driver_trainingvalidation2/extra_driver_trainingvalidation2-cluster3.pickle",
            ],
            "Rturn_industrialrc_asphaltc":["Rturn_industrialrc_asphaltc/Rturn_industrialrc_asphaltc-cluster8.pickle",
                                           "Rturn_industrialrc_asphaltc/Rturn_industrialrc_asphaltc-cluster9.pickle",
            ],
            "Rturn_servicecutthru": ["Rturn_servicecutthru/Rturn_servicecutthru-cluster9.pickle",
                                     "Rturn_servicecutthru/Rturn_servicecutthru-cluster3.pickle",
                                     "Rturn_servicecutthru/Rturn_servicecutthru-cluster8.pickle",
                                     "Rturn_servicecutthru/Rturn_servicecutthru-cluster5.pickle"
            ],
            "Rturnserviceroad": ["Rturnserviceroad/Rturnserviceroad-cluster6.pickle", 
                                 "Rturnserviceroad/Rturnserviceroad-cluster7.pickle"
            ],
            "Rturncommercialunderpass": ["Rturncommercialunderpass/Rturncommercialunderpass-cluster1.pickle", 
                                         "Rturncommercialunderpass/Rturncommercialunderpass-cluster5.pickle"
            ],



            "extra_windingnarrowtrack": ["extra_windingnarrowtrack/extra_windingnarrowtrack-cluster6.pickle",
                                         "extra_windingnarrowtrack/extra_windingnarrowtrack-cluster3.pickle"
            ],
            "extra_windingtrack": ["extra_windingtrack/extra_windingtrack-cluster0.pickle", 
                                   "extra_windingtrack/extra_windingtrack-cluster9.pickle", 
                                   "extra_windingtrack/extra_windingtrack-cluster7.pickle"
            ],
            "extra_utahlong": ["extra_utahlong/extra_utahlong-cluster2.pickle", 
                               "extra_utahlong/extra_utahlong-cluster5.pickle"
            ],
            "extra_westunderpasses": ["extra_westunderpasses/extra_westunderpasses-cluster3.pickle"],
            "extra_jungledrift_road_d":["extra_jungledrift_road_d/extra_jungledrift_road_d-cluster7.pickle",
                                        "extra_jungledrift_road_d/extra_jungledrift_road_d-cluster8.pickle"
            ],
            "extra_jungledrift_road_e": ["extra_jungledrift_road_e/extra_jungledrift_road_e-cluster2.pickle", 
                                         "extra_jungledrift_road_e/extra_jungledrift_road_e-cluster8.pickle"
            ],
            "extra_jungledrift_road_s":["extra_jungledrift_road_s/extra_jungledrift_road_s-cluster7.pickle", # might be too high dev
                                        "extra_jungledrift_road_s/extra_jungledrift_road_s-cluster9.pickle"
             ],
            "extra_wideclosedtrack": ["extra_wideclosedtrack/extra_wideclosedtrack-cluster3.pickle",
                                      "extra_wideclosedtrack/extra_wideclosedtrack-cluster8.pickle",
                                      "extra_wideclosedtrack/extra_wideclosedtrack-cluster5.pickle",
            ],
            "extra_dock": ["extra_dock/extra_dock-cluster2.pickle",
                           "extra_dock/extra_dock-cluster4.pickle",
                           "extra_dock/extra_dock-cluster0.pickle"
            ],
            "extrawinding_industrial7978":["extrawinding_industrial7978/extrawinding_industrial7978-cluster0.pickle",
                                            "extrawinding_industrial7978/extrawinding_industrial7978-cluster3.pickle"
            ],
            "straightcommercialroad":["straightcommercialroad/straightcommercialroad-cluster7.pickle"]
}

BASERUN_ROOT = "/p/sdbb/safetransf-base-model-validation-tracks-ALL/safetransf-base-model-validation-tracks/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051-7ZYMOA/"
road_ids = ["Lturnpasswarehouse",
            "extra_driver_trainingvalidation2",
            "Rturn_industrialrc_asphaltc",
            "Rturn_servicecutthru",
            "Rturnserviceroad",
            "Rturncommercialunderpass",
            "extra_windingnarrowtrack",
            "extra_windingtrack",
            "extra_utahlong",
            "extra_westunderpasses",
            "extra_jungledrift_road_d",
            "extra_jungledrift_road_e",
            "extra_jungledrift_road_s",
            "extra_wideclosedtrack",
            "extra_dock",
            "extrawinding_industrial7978",
            "straightcommercialroad",
]


def plot_deviation(trajectories, centerline, left, right, xlim=None, ylim=None, resultsdir="images", topo_id=None, title=None):
    fig, ax = plt.subplots()
    for i, trajectory in enumerate(trajectories):
        x = [point[0] for point in trajectory]
        y = [point[1] for point in trajectory]
        plt.plot(x, y, label=f"Traj {i}", linewidth=2)
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

def randstr():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))


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

def lineseg_dists2(p,a,b):
    d=np.cross(b-a,p-a)/np.linalg.norm(b-a)
    return d

@ignore_warnings
def dist_from_line(centerline, point):
    a = [[x[0], x[1]] for x in centerline[:-1]]
    b = [[x[0], x[1]] for x in centerline[1:]]
    a = np.array(a)
    b = np.array(b)
    # print(f"{a.shape=}")
    # print(f"{b.shape=}")
    dist = lineseg_dists([point[0], point[1]], a, b)
    # print(f"{dist=}")
    return dist

@ignore_warnings
def get_nearest_point(line, point):
    # print(f"{np.array(line).shape=}")
    dists = dist_from_line(line, point)
    # print(f"{dists=}")
    index = np.nanargmin(np.array(dists))
    # print(f"{index} \t{line[index]}")
    line_point = line[index]
    return line_point, index


def distance2D(a,b):
    return math.sqrt(math.pow(a[0]- b[0], 2) + math.pow(a[1] - b[1], 2))

@ignore_warnings
def get_deviation(trajectory, centerline):
    dists = []
    for point in trajectory:
        try:
            dist = dist_from_line(centerline, point)
        except:
            print(f"centerline shape={np.array(centerline).shape}")
            print(f"point shape={np.array(point).shape}")
            exit(0)
        dists.append(np.nanmin(dist))
    return np.mean(dists)


def get_rot_quat(angle):
    r = R.from_euler("z", angle, degrees=True)
    return r.as_quat()


reverses = ["extra_driver_trainingvalidation2", "extra_westcoastrocks"]
clusters_of_interest = {"Rturn_servicecutthru-cluster3": f"{CLUSTER_ROOT}/Rturn_servicecutthru/Rturn_servicecutthru-cluster3.pickle", # 7 3 9
                        "Lturnpasswarehouse-cluster7": f"{CLUSTER_ROOT}/Lturnpasswarehouse/Lturnpasswarehouse-cluster7.pickle",
                        "extra_windingtrack-cluster9": f"{CLUSTER_ROOT}/extra_windingtrack/extra_windingtrack-cluster9.pickle", # 1 3 6 9
                        "extra_westunderpasses-cluster3": f"{CLUSTER_ROOT}/extra_westunderpasses/extra_westunderpasses-cluster3.pickle",
                        "extra_utahlong-cluster2": f"{CLUSTER_ROOT}/extra_utahlong/extra_utahlong-cluster2.pickle",
                        "Rturn_industrialrc_asphaltc-cluster9" : f"{CLUSTER_ROOT}/Rturn_industrialrc_asphaltc/Rturn_industrialrc_asphaltc-cluster9.pickle",
                        "extrawinding_industrial7978-cluster3" : f"{CLUSTER_ROOT}/extrawinding_industrial7978/extrawinding_industrial7978-cluster3.pickle",
                        "extrawinding_industrial7978-cluster0" : f"{CLUSTER_ROOT}/extrawinding_industrial7978/extrawinding_industrial7978-cluster0.pickle",
                        "extra_dock-cluster2" : f"{CLUSTER_ROOT}/extra_dock/extra_dock-cluster2.pickle",
}

# simrun_results.keys()=
# dict_keys(['trajectories', 'dists_from_centerline', 'dists_travelled', 'track_length', 'img_dims', 'centerline_interpolated', 
# 'roadleft', 'roadright', 'default_scenario', 'road_id', 'topo_id', 'transf_id', 'vqvae_name', 'vqvae_id', 'model_name', 'runtimes'])
# cluster_results.keys()=
# dict_keys(['roadleft', 'roadright', 'centerline', 'trajectory', 'deviation', 'traj_len', 'ctrline_length'])
import re
def main():
    simresultsdir = "/p/sdbb/results-09-05-2023"
    newresultsdir = f"./parsedsimresults/parsed_{simresultsdir.split('/')[-1]}_" + randstr()
    os.makedirs(newresultsdir, exist_ok=True)
    clusterwise_results = dict.fromkeys(clusters_of_interest.keys())
    for cl_id in clusterwise_results.keys():
        print(f"clusterwise results[{cl_id}]")
        clusterwise_results[cl_id] = {"trajectories": [],
                                        "deviations": [],
                                        "dist_travelled": [],
                                        "distances": [],
                                        # "":,
                                    }
    subDirs1 = [f"{simresultsdir}/{_}" for _ in os.listdir(simresultsdir) if os.path.isdir(f"{simresultsdir}/{_}")]
    subDirs1.sort()
    for d1 in subDirs1[12:]:
        subDirs2 = [f"{d1}/{_}" for _ in os.listdir(f"{d1}") if os.path.isdir(f"{d1}/{_}")]
        subDirs2.sort()
        all_distances, all_deviations, all_track_lengths = [], [], []
        for d2 in subDirs2:
            print()
            simrun_results = unpickle_results(f"{d2}/summary.pickle")
            cluster =  re.search('cluster[0-9]', d2)
            cl = clusters_of_interest[simrun_results["topo_id"] + "-" + cluster.group(0)]
            cluster_results = unpickle_results(cl)
            # print(d2, simrun_results["topo_id"])
            if True: # simrun_results["topo_id"] == "extra_windingtrack": # Rturn_servicecutthru extra_windingtrack 
                centerline_full = simrun_results["centerline_interpolated"]
                centerline_seg = cluster_results["centerline"]
                # print(f"{cl}     {len(centerline_seg)=:.1f}")
                trajectories, deviations, distances = [], [], []
                for t in simrun_results["trajectories"]:
                    # print(d2)
                    t = [[p[0], p[1]] for p in t]
                    t_point_start, t_index_start = get_nearest_point(t, centerline_seg[0])
                    t_point_end, t_index_end = get_nearest_point(t, centerline_seg[-1])
                    if t_index_start > t_index_end:
                        temp = t_index_start
                        t_index_start = t_index_end
                        t_index_end = temp
                    t_seg = t[t_index_start:t_index_end]
                    # print(f"{t_index_start=} \t{t_index_end=}")
                    trajectories.append(t_seg)
                    deviation = get_deviation(t_seg, centerline_seg)
                    deviations.append(deviation)
                    dist_travelled = sum([distance2D(a,t_seg[i+1]) for i,a in enumerate(t_seg[:-1])])
                    distances.append(dist_travelled)
                # clusterwise_results[cl["topo_id"] + "_" + cl["cluster_id"]] = {"trajectories": trajectories,
                #                                                             "deviations": deviations,
                #                                                             "dist_travelled": dist_travelled,
                #                                                             "distances": distances,
                #                                                             # "":,
                #                                                         }
                # calculate distance and deviation
                # print(distances, dist_travelled, deviations)
                track_seg_len = sum([distance2D(a,centerline_seg[i+1]) for i,a in enumerate(centerline_seg[:-1])])
                meandist = np.nanmean(distances)
                stddist = np.nanstd(distances)
                meandev = np.nanmean(deviations)
                all_distances.append(meandist)
                all_deviations.extend(deviations)
                all_track_lengths.append(track_seg_len)

                clusterwise_results[cl["topo_id"] + "_" + cl["cluster_id"]]["trajectories"].append(trajectories)
                # clusterwise_results[cl["topo_id"] + "_" + cl["cluster_id"]]["dist_travelled"].extend(distances)
                clusterwise_results[cl["topo_id"] + "_" + cl["cluster_id"]]["deviations"].extend(deviations)
                clusterwise_results[cl["topo_id"] + "_" + cl["cluster_id"]]["distances"].extend(distances)
                clusterwise_results[cl["topo_id"] + "_" + cl["cluster_id"]]["track_len"] = track_seg_len

                print(f"{d2.split('/')[-1]}"
                        f"\navg. distance trav.=\t\t{meandist:.1f}"
                        f"\ndev. distance trav.=\t\t{stddist:.1f}"
                        f"\navg dist. from ctr.=\t\t{meandev:.3f}"
                        f"\ntrack  seg.  length=\t\t{track_seg_len:.1f}")

                title = f"{d2.split('/')[-1]}"
                title = " ".join(title.split("-")[:-2])
                title = title + f"\navg. distance trav.={meandist:.1f} dev. distance trav.={stddist:.1f} avg dist. from ctr.={meandev:.3f}"
                plot_deviation(trajectories, centerline_seg, cluster_results["roadleft"], cluster_results["roadright"], 
                                resultsdir=newresultsdir, topo_id=simrun_results["topo_id"], title=title)
        print(f"\n{len(all_distances)} TRACK SUMMARY {d1}"
                f"\navg. distance trav.=\t\t{np.nanmean(all_distances):.1f}"
                f"\ndev. distance trav.=\t\t{np.nanstd(all_distances):.1f}"
                f"\navg dist. from ctr.=\t\t{np.nanmean(all_deviations):.3f}"
                f"\navg.  track  length=\t\t{np.nanmean(all_track_lengths):.1f}")
        for cl_id in clusterwise_results.keys():
            print(f"4-TRANSF SUMMARY FOR {cl_id}"
                  f"\navg. distance trav.=\t\t{np.nanmean(clusterwise_results[cl_id]['distances']):.1f}"
                  f"\ndev. distance trav.=\t\t{np.nanstd(clusterwise_results[cl_id]['distances']):.1f}"
                  f"\navg dist. from ctr.=\t\t{np.nanmean(clusterwise_results[cl_id]['deviations']):.3f}"
                  f"\ntrack  length=\t\t{clusterwise_results[cl_id]['track_len']:.1f}"
                  f"\n"
                )
            
    print(f"Results written to {newresultsdir}")
                

if __name__ == "__main__":
    main()