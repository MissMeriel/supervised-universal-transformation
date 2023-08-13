import numpy as np
from matplotlib import pyplot as plt
import logging, random, string, time, os
from tabulate import tabulate
from pathlib import Path
import argparse, sys
import statistics, math
from ast import literal_eval
from PIL import Image
import pickle
import itertools
from tqdm import tqdm
import pandas as pd
import re
import itertools
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", '--dataset', type=str, default="../data", help='parent directory of results dataset')
    parser.add_argument('-f', '--figure', type=str, help='table or figure to generate (choose from: table1 table2 table3 table5 table6 figure5 figure6 figure7 figure8)')
    args = parser.parse_args()
    return args


def unpickle_results(filename):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results


def process_rough(parentdir):
    fileExt = r".pickle"
    allDirs = [f"{parentdir}/{_}" for _ in os.listdir(f"{parentdir}") if
               os.path.isdir("/".join([parentdir,  _]))]
    keys = [
        "results-dbb-orig-5-1000-400-",
        "results-dbb-orig-10-1000-400-",
        "results-dbb-orig-15-1000-400-"
    ]
    results_map = {}

    for key in keys:

        bullseyes, crashes, goals = [], [], []
        aes, dists = [], []
        num_bbs, MAE_coll, ttrs = [], [], []
        for dirs in tqdm(allDirs):


            if key in dirs:
                results_files = [_ for _ in os.listdir(dirs) if _.endswith(fileExt)]
                for file in results_files:
                    results_file = "/".join([dirs, file])
                    print(results_file)
                    results = unpickle_results(results_file)
                    outcomes_percents = get_outcomes(results)
                    bullseyes.append(outcomes_percents['B'])
                    crashes.append(outcomes_percents['D'] + outcomes_percents['LT'] + outcomes_percents['B'])
                    goals.append(outcomes_percents['GOAL'])
                    aes.extend(results['testruns_errors'])
                    dists.extend(results['testruns_dists'])
                    ttrs.append(results['time_to_run_technique'])
                    num_bbs.append(results['num_billboards'])
                    MAE_coll.append(results['MAE_collection_sequence'])
            print(key)
            print(f"{len(crashes)=}")
            print(f"{len(dists)=}")
            print(f"{len(aes)=}")
            print(f"{len(MAE_coll)=}")
            results_map[key] = {"bullseyes": bullseyes, "crashes": crashes, "goals": goals,
                                "aes": aes, "dists": dists, "ttrs": ttrs, "num_bbs": num_bbs, "MAE_coll": MAE_coll}

    listified_map = [[key, f"{sum(results_map[key]['crashes']) / (10 * len(results_map[key]['crashes'])):.3f}",
                      f"{round(sum(results_map[key]['dists']) / len(results_map[key]['dists']), 4)}",
                      f"{round(sum(results_map[key]['aes']) / len(results_map[key]['aes']), 3)}",
                      f"{sum(results_map[key]['MAE_coll']) / len(results_map[key]['MAE_coll']):.3f}",
                      f"{len(results_map[key]['crashes'])}"] for key in results_map.keys()]

    print(tabulate(listified_map, headers=["technique", "crash rate", "dist from exp traj", "AAE", "expected AAE", "samples"], tablefmt="github"))

if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    args = parse_arguments()
    parentdir = "E:/GitHub/DeepManeuver/tools/simulation/results/RERUNMISSING-scenario4-5LWNZY"
    process_rough(parentdir)