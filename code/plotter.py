'''
This file automates the generation of a plot for the methods used.
'''

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import OrderedDict


linestyles = [
    (0, ()),
    (0, (1, 1)),

    (0, (5, 10)),
    (0, (5, 5)),

    (0, (3, 1, 1, 1)),
    (0, (3, 10, 1, 10)),
    (0, (5, 1)),
    (0, (3, 5, 1, 5)),

    (0, (3, 10, 1, 10, 1, 10)),
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (1, 10)),
    (0, (1, 5)),
]


colors = [
    'green',
    'navy',
    'dodgerblue',
    'orange',
    'black',
    'black',
    'black',
    'black',
    'black',
    'black',
    'black',
    'black',
    'black',
]


if sys.version_info[0] == 2:
    import cPickle as pickle
elif sys.version_info[0] == 3:
    import _pickle as pickle
else:
    raise Exception("Error - Unknown python version!")

KT300_to_KJMOL = 2.4942796


def plot(stats, se_scale=2.0, keys=["PAL_0", "PAL", "SIMPLE", "SMAC", "SMAC_ORD", "RANDOM", "HUTTER"], aliases={}):
    '''
    This function automates the plotting of the image in our paper.
    '''

    for key in keys:
        if key not in aliases:
            aliases[key] = key

    # Larger image
    plt.figure(figsize=(10, 8))

    # We store the keys to ensure the order is always the same.
    offset = {"PAL_0": 0, "PAL": 0, "SMAC": 0, "SMAC_ORD": 0, "SIMPLE": 0, "RANDOM": 0, "HUTTER": 0}
    ls = {k: linestyles[i] for i, k in enumerate(keys)}
    lc = {k: colors[i] for i, k in enumerate(keys)}
    # keys = ["PAL", "SIMPLE", "SMAC", "SMAC_ORD", "RANDOM"]
    # offset = {"PAL": 0, "SMAC": 0, "SMAC_ORD": 0, "SIMPLE": 0, "RANDOM": 0}
    for key in keys:
        N_POINTS = len(stats[key][0])
        mu, se = stats[key]
        mu = mu * -1.0
        mu = mu[offset[key]:]
        se = se[offset[key]:]

        # Unit conversion of kT_300 to kj/mol
        mu = mu * KT300_to_KJMOL
        se = se * KT300_to_KJMOL

        plt.plot(range(offset[key], N_POINTS), mu, linestyle=ls[key], color=lc[key], label=aliases[key], linewidth=3)
        plt.fill_between(range(offset[key], N_POINTS), mu - se_scale * se, mu + se_scale * se, color=lc[key], alpha=0.5)

    plt.legend(bbox_to_anchor=(0.99, 0.3), fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlim([0, 60])
    plt.ylim([-42 * KT300_to_KJMOL, -15 * KT300_to_KJMOL])

    # plt.xlim([1, 240])
    # plt.ylim([-41.35 * KT300_to_KJMOL, -41 * KT300_to_KJMOL])

    # plt.xlim([1, 50])
    # plt.ylim([-42 * KT300_to_KJMOL, -37 * KT300_to_KJMOL])

    plt.xlabel("Number of Observations", fontsize=16)
    plt.ylabel("Intermolecular Binding Energy (kJ/mol)", fontsize=16)

    plt.gca().invert_yaxis()

    # Make a section for our training set region (if we had one)
    if offset["PAL_0"] > 0:
        xvals = np.arange(0, offset["PAL_0"] + 1)
        yvals1 = [10 * KT300_to_KJMOL for x in xvals]
        yvals2 = [-100 * KT300_to_KJMOL for x in xvals]
        plt.fill_between(xvals, yvals1, yvals2, facecolor='gray', alpha=0.3)
        plt.text(0.4, -16 * KT300_to_KJMOL, "PAL_0 Training\n    Region", fontsize=12)

    if not os.path.exists("out"):
        os.mkdir("out")

    # plt.show()
    plt.savefig("out/bench.png")


if __name__ == "__main__":
    a, _ = pickle.load(open("../data/final.pickle", 'rb'))
    # a, _ = pickle.load(open("out/final2.pickle", 'rb'))
    keys = ["PAL", "SIMPLE", "HUTTER", "SMAC", "RANDOM"]
    aliases = {"SIMPLE": "Simple BO", "SMAC": "pySMAC", "HUTTER": "Hutter BO", "RANDOM": "Random"}
    plot(a, keys=keys, aliases=aliases)
