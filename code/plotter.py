'''
This file automates the generation of a plot for the methods used.
'''

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    import cPickle as pickle
elif sys.version_info[0] == 3:
    import _pickle as pickle
else:
    raise Exception("Error - Unknown python version!")

KT300_to_KJMOL = 2.4942796


def plot(stats, se_scale=2.0):
    '''
    This function automates the plotting of the image in our paper.
    '''

    # Larger image
    plt.figure(figsize=(10, 8))

    # We store the keys to ensure the order is always the same.
    keys = ["PAL", "SMAC", "SIMPLE", "RANDOM"]
    offset = {"PAL": 0, "SMAC": 0, "SIMPLE": 0, "RANDOM": 0}
    for key in keys:
        N_POINTS = len(stats[key][0])
        mu, se = stats[key]
        mu = mu * -1.0
        mu = mu[offset[key]:]
        se = se[offset[key]:]

        # Unit conversion of kT_300 to kj/mol
        mu = mu * KT300_to_KJMOL
        se = se * KT300_to_KJMOL

        plt.plot(range(offset[key], N_POINTS), mu, label=key, linewidth=3)
        plt.fill_between(range(offset[key], N_POINTS), mu - se_scale * se, mu + se_scale * se, alpha=0.5)

    plt.legend(bbox_to_anchor=(1, 0.15), fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlim([1, 50])
    plt.ylim([-42 * KT300_to_KJMOL, -37 * KT300_to_KJMOL])

    plt.xlabel("Number of Observations", fontsize=16)
    plt.ylabel("Intermolecular Binding Energy (kJ/mol)", fontsize=16)

    plt.gca().invert_yaxis()

    # Make a section for our training set region (if we had one)
    if offset["PAL"] > 0:
        xvals = np.arange(0, offset["PAL"] + 1)
        yvals1 = [10 * KT300_to_KJMOL for x in xvals]
        yvals2 = [-100 * KT300_to_KJMOL for x in xvals]
        plt.fill_between(xvals, yvals1, yvals2, facecolor='gray', alpha=0.3)
        plt.text(2, -40 * KT300_to_KJMOL, "PAL Training\n    Region")

    if not os.path.exists("out"):
        os.mkdir("out")

    plt.savefig("out/bench.png")


if __name__ == "__main__":
    a, _ = pickle.load(open("final.pickle", 'rb'))
    plot(a)
