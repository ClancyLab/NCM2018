'''
This is the main code that should automate the process of re-running the
benchmarks using PAL, pySMAC, and pure randomness.  Helper codes are
imported and run, and a final plotting code is run to illustrate the final
results.  Image from the paper is generated when NUM_RUNS is set to defaults.

REQUIREMENTS:
---------------------------------
0. Ensure you have python 3 installed with numpy and matplotlib
    NOTE! This will still work for python 2, but pySMAC will timeout (not an
    error, but you'll get annoying warning outputs during runtime)
1. Install pySMAC (pip install git+https://github.com/automl/pysmac.git --user)
'''
import os
import sys

from run_pal import run_pal
# from run_pal_0 import run_pal_0
from run_simple import run_simple
from run_hutter import run_hutter
from run_pysmac import run_pysmac
# from run_pysmac_ord import run_pysmac_ord
from run_random import run_random
from plotter import plot
from statter import pretty_stats, pretty_stats_2

if sys.version_info[0] == 2:
    import cPickle as pickle
elif sys.version_info[0] == 3:
    import _pickle as pickle
else:
    raise Exception("Error - Unknown python version!")

assert not os.path.exists("out"), "Error - Will not run whilst the out folder exists."
os.mkdir("out")

dataset, stats = {}, {}
dataset, stats = run_pal(dataset, stats, NUM_RUNS=10, parallel=False, on_queue=False)  # Default, NUM_RUNS = 1,000
dataset, stats = run_simple(dataset, stats, NUM_RUNS=10, parallel=False, on_queue=False)  # Default, NUM_RUNS = 1,000
dataset, stats = run_hutter(dataset, stats, NUM_RUNS=10, parallel=False, on_queue=False)  # Default, NUM_RUNS = 1,000
dataset, stats = run_pysmac(dataset, stats, NUM_RUNS=10, on_queue=False)  # Default, NUM_RUNS = 1,000
dataset, stats = run_random(dataset, stats, NUM_RUNS=10000)  # Default, NUM_RUNS = 1,000,000

print("\nAll Done!")
print("-------------------------------------------------")

pickle.dump([dataset, stats], open("out/final.pickle", 'wb'))
keys = ["PAL", "SIMPLE", "HUTTER", "SMAC", "RANDOM"]
aliases_plot = {"SIMPLE": "Simple BO", "SMAC": "pySMAC", "HUTTER": "Hutter BO", "RANDOM": "Random"}
aliases_stat = {"SIMPLE": "Simple_BO", "SMAC": "pySMAC", "HUTTER": "Hutter_BO", "RANDOM": "Random"}

plot(dataset, keys=keys, aliases=aliases_plot)
pretty_stats(stats, keys=keys, aliases=aliases_stat)
pretty_stats_2(stats, keys=keys, aliases=aliases_stat)

print("-------------------------------------------------")
