import cPickle as pickle

# Note - to import this you'll either have to be in the same folder as plotter
# and portable_pal, or add that path to your PYTHONPATH environment variable.
from plotter import plot
from statter import pretty_stats, pretty_stats_2
from portable_pal import parseNum

# Read in the actual DFT calculation results and corresponding combination.
objs = pickle.load(open("../data/all_data.pickle", 'r'))

# Print out all the DFT data
for obj in objs:
    print("--------------------------------------------------------")
    print("Combination: %s" % parseNum(obj[:-1]))
    print("The x descriptor for this combination is: %s" % str(obj[:-1]))
    print("The associated intermolecular binding energy is: %.4f" % float(obj[-1]))
print("--------------------------------------------------------")

# Read in the "stats" data.  This holds the mean and the standard error of measurement.
data, stats = pickle.load(open("../data/final.pickle", 'r'))

# Plot it if you want
plot(data)
pretty_stats(stats)
pretty_stats_2(stats)
