'''
'''

import re
import sys
import numpy as np

if sys.version_info[0] == 2:
    import cPickle as pickle
elif sys.version_info[0] == 3:
    import _pickle as pickle
else:
    raise Exception("Error - Unknown python version!")


def _spaced_print(sOut, delim=['\t', ' '], buf=4):
    """
    Given a list of strings, or a string with new lines, this will reformat the string with spaces
    to split columns.  Note, this only works if there are no headers to the input string/list of strings.

    **Parameters**

        sOut: *str* or *list, str*
            String/list of strings to be formatted.
        delim: *list, str*
            List of delimiters in the input strings.
        buf: *int*
            The number of spaces to have between columns.

    **Returns**

        spaced_s: *str*
            Appropriately spaced output string.
    """
    s_len = []
    if type(sOut) == str:
        sOut = sOut.split('\n')
    if type(delim) == list:
        delim = ''.join([d + '|' for d in delim])[:-1]
    # Get the longest length in the column
    for i, s in enumerate(sOut):
        s = re.split(delim, s)
        for j, ss in enumerate(s):
            try:
                # This makes the part of the list for each column the longest length
                s_len[j] = len(ss) if len(ss) > s_len[j] else s_len[j]
            except:
                # If we are creating a new column this happens
                s_len.append(len(ss))
    # Now we add a buffer to each column
    for i in range(len(s_len)):
        s_len[i] += buf

    # Compile string output
    for i, s in enumerate(sOut):
        s = re.split(delim, s)
        for j, ss in enumerate(s):
            s[j] = ss + ''.join([' '] * (s_len[j] - len(ss)))
        sOut[i] = ''.join(s)

    return '\n'.join(sOut)


def pretty_stats(stats, p=0.999, keys=["PAL_0", "PAL", "HUTTER", "SIMPLE", "SIMPLE_0", "SMAC", "SMAC_ORD", "RANDOM"], aliases={}):
    for key in keys:
        if key not in aliases:
            aliases[key] = key
    data = "Methods\tAvg\tStd\t%.1f%%\tReplications\n" % (p * 100.0)
    for key in keys:
        avg = np.mean(stats[key][0])
        std = np.std(stats[key][0])
        reps = len(stats[key][0])
        p95 = int(len(stats[key][0]) * p - 1.0)
        p95 = sorted(stats[key][0])[p95]
        data += "%s\t%.1f\t%.1f\t%d\t%d\n" % (aliases[key], avg, std, p95, reps)

    data = _spaced_print(data).split("\n")
    data = [data[0]] + ["-" * len(data[0])] + data[1:]
    print("\n")
    print('\n'.join(data))


def pretty_stats_2(stats, p=[0.95, 0.98, 0.999], keys=["PAL_0", "PAL", "HUTTER", "SIMPLE", "SIMPLE_0", "SMAC", "SMAC_ORD", "RANDOM"], aliases={}):
    for key in keys:
        if key not in aliases:
            aliases[key] = key
    data = "Methods\t" + '\t'.join(["%.1f%%" % (pp * 100) for pp in p]) + '\n'
    for key in keys:
        pval = [int(len(stats[key][0]) * pp - 1.0) for pp in p]
        pval = [sorted(stats[key][0])[pp] for pp in pval]
        data += str(aliases[key]) + "\t" + "\t".join(["%d" % pp for pp in pval]) + "\n"

    data = _spaced_print(data).split("\n")
    data = [data[0]] + ["-" * len(data[0])] + data[1:]
    print("\n")
    print('\n'.join(data))


if __name__ == "__main__":
    _, a = pickle.load(open("out/final2.pickle", 'rb'))
    keys = ["PAL", "SIMPLE", "HUTTER", "SMAC", "RANDOM"]
    aliases = {"SIMPLE": "Simple_BO", "SMAC": "pySMAC", "HUTTER": "Hutter_BO", "RANDOM": "Random"}
    pretty_stats(a, keys=keys, aliases=aliases)
    pretty_stats_2(a, keys=keys, aliases=aliases)
