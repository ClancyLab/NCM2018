import sys
import copy
import scipy.stats
import numpy as np
import portable_pal as pal

if sys.version_info[0] == 2:
    import cPickle as pickle
elif sys.version_info[0] == 3:
    import _pickle as pickle
else:
    raise Exception("Error - Unknown python version!")


def run_random(dataset, stats, NUM_RUNS=1000):
    # -------------------------------------------------------------------------------------------------
    # Ex. 1 - Random

    # Read in all the data points
    data = pickle.load(open("all_data.pickle", 'rb'))
    data_y = [row[-1] for row in data]
    N_POINTS = len(data_y)

    best_y = []
    n_to_max = []
    for i in range(NUM_RUNS):
        pal.printProgressBar(i, NUM_RUNS, prefix="Running RANDOM case...")
        this_run = copy.deepcopy(data_y)
        np.random.shuffle(this_run)
        this_run = [max(this_run[:i + 1]) if i > 0 else x for i, x in enumerate(this_run)]
        best_y.append(this_run)
        n_to_max.append(this_run.index(max(this_run)))
    best_y = np.array(best_y)

    dataset["RANDOM"] = [
        np.array([np.mean(best_y[:, i]) for i in range(N_POINTS)]),
        np.array([scipy.stats.sem(best_y[:, i]) for i in range(N_POINTS)])
    ]

    pickle.dump(dataset["RANDOM"], open("out/d_random.dat", 'wb'))
    pickle.dump(best_y, open("out/best_random.dat", 'wb'))

    stats["RANDOM"] = [n_to_max, N_POINTS]

    pal.printProgressBar(NUM_RUNS, NUM_RUNS, prefix="Running RANDOM case...", suffix="Done")
    return dataset, stats


if __name__ == "__main__":
    print("Running RANDOM for only one replication! If you wish to do a full run, please use the run.py code.")
    dataset, stats = run_random({}, {}, NUM_RUNS=1)
