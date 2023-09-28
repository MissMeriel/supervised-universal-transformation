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
Script to generate the config files for in-simulation validation runs
on 100m segments of track
'''


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
                                     "Rturn_servicecutthru/Rturn_servicecutthru-cluster7.pickle",
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
                                   "extra_windingtrack/extra_windingtrack-cluster3.pickle", 
                                   "extra_windingtrack/extra_windingtrack-cluster6.pickle", 
                                   "extra_windingtrack/extra_windingtrack-cluster7.pickle",
                                   "extra_windingtrack/extra_windingtrack-cluster9.pickle",
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


def dist_from_line(centerline, point):
    a = [[x[0], x[1]] for x in centerline[:-1]]
    b = [[x[0], x[1]] for x in centerline[1:]]
    a = np.array(a)
    b = np.array(b)
    dist = lineseg_dists([point[0], point[1]], a, b)
    return dist


def get_nearest_point(line, point):
    dists = dist_from_line(line, point)
    index = np.nanargmin(np.array(dists))
    line_point = line[index]
    return line_point, index


def distance2D(a,b):
    return math.sqrt(math.pow(a[0]- b[0], 2) + math.pow(a[1] - b[1], 2))


def get_rot_quat(angle):
    r = R.from_euler("z", angle, degrees=True)
    return r.as_quat()

reverses = ["extra_driver_trainingvalidation2", "extra_westcoastrocks"]

def main():
    config_filename = "./config-segments/config-segments_" + randstr() + ".csv"
    with open(config_filename, "w") as f:
        f.write("TOPOID,SEGNUM,START,END,SPAWN,CUTOFF,DEGREES,ROT_QUAT\n")
        for ck in cluster_paths.keys():
            clusters = cluster_paths[ck]
            # if True: # 
            for c in clusters:
                # c = clusters[0]
                # unpickle
                cluster_results = unpickle_results(CLUSTER_ROOT + c)
                extradir = [filename for filename in os.listdir(BASERUN_ROOT + ck)][0]
                pck = [filename for filename in os.listdir(BASERUN_ROOT + ck + "/" + extradir) if filename.endswith(".pickle")][0]
                baserun_results = unpickle_results(BASERUN_ROOT + ck + "/" + extradir + "/" + pck)
                centerline_full = baserun_results["centerline_interpolated"]
                centerline_seg = cluster_results["centerline"]
                # find position and rotation
                ctrline_point_start, ctrline_index_start = get_nearest_point(centerline_full, centerline_seg[0])
                ctrline_point_end, ctrline_index_end = get_nearest_point(centerline_full, centerline_seg[-1])
                ctrline_point_end_plusmargin = centerline_full[np.min([ctrline_index_end + 20, len(centerline_full)-1])]
                uptrack_start = centerline_full[ctrline_index_start - 20]
                uptrack_angle_pos = centerline_full[ctrline_index_start - 17]
                # angle = math.atan2(ctrline_point_start[1] - uptrack_start[1], ctrline_point_start[0] - uptrack_start[0])
                angle = math.atan2(-(uptrack_angle_pos[0] - uptrack_start[0]), (uptrack_angle_pos[1] - uptrack_start[1])) - math.pi / 2
                # print(f"{ctrline_index_start=} \t{ctrline_index_end=}")
                # exit(0)
                if ck in reverses:
                    # reverse ctrline seg and reverse start and end indices
                    print(ck, "Must reverse centerline seg ")
                    temp = ctrline_index_start
                    ctrline_index_start = ctrline_index_end
                    ctrline_index_end = temp
                    temp = ctrline_point_start
                    ctrline_point_start = ctrline_point_end
                    ctrline_point_end = temp
                    ctrline_point_end_plusmargin = centerline_full[np.max([ctrline_index_end - 20, 0])]
                    # uptrack_start = centerline_full[np.min([ctrline_index_start + 20, len(centerline_full)-1])]
                    # uptrack_angle_pos = centerline_full[np.min([ctrline_index_start + 17, len(centerline_full)-1])]
                    uptrack_start = centerline_full[ctrline_index_start + 20]
                    uptrack_angle_pos = centerline_full[ctrline_index_start + 17]
                    # angle = math.atan2((uptrack_start[1] - uptrack_angle_pos[1]), -(uptrack_start[0] - uptrack_angle_pos[0])) - math.pi / 2
                    angle = math.atan2(-(uptrack_start[0] - uptrack_angle_pos[0]), (uptrack_start[1] - uptrack_angle_pos[1])) - math.pi / 2

                rot_quat = get_rot_quat(math.degrees(angle))
                segnum = c.replace(".pickle","").split("-")[-1]
                segnum = segnum.replace("cluster","")
                f.write(f"{ck},{segnum},{str(ctrline_point_start).replace(',', ' ')},{str(ctrline_point_end).replace(',', ' ')},{str(uptrack_start).replace(',', ' ')},{str(ctrline_point_end_plusmargin).replace(',', ' ')},{math.degrees(angle)},{str(rot_quat).replace(',', ' ')}\n")
    print(f"Results written to {config_filename}")
                

if __name__ == "__main__":
    main()