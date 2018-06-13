'''
This is a helper code to combine an output directory into a single pickle file
for plotter.py and statter.py to analyze.  Note, this should be automatically
run by default in run.py, but we include it just in case of a bug.
'''

import os
import cPickle as pickle
import numpy as np
import scipy.stats

folder = "run_2"

N_POINTS = 240

dataset = {}
stats = {}
for fptr in os.listdir(folder):
    if not fptr.endswith(".dat"):
        continue
    if not fptr.startswith("best_"):
        continue
    name = fptr.split("_")[1].split(".")[0].upper()

    each_run = pickle.load(open("%s/%s" % (folder, fptr)))

    dataset[name] = [
        np.array([np.mean(each_run[:, i]) for i in range(N_POINTS)]),
        np.array([scipy.stats.sem(each_run[:, i]) for i in range(N_POINTS)])
    ]

    n_to_max = []
    for this_run in each_run.tolist():
        n_to_max.append(this_run.index(max(this_run)))

    stats[name] = [n_to_max, N_POINTS]

pickle.dump([dataset, stats], open("combined.pickle", 'w'))
