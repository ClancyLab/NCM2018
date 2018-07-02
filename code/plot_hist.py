import cPickle as pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))

# Only use the following line if you have enough ram to handle the random data
# names = ["random", "smac_ord", "smac", "hutter", "simple", "pal", "pal_0"]
names = ["pal", "simple", "hutter", "smac", "random"]
aliases = {"simple": "Simple BO", "smac": "pySMAC", "hutter": "Hutter BO", "pal": "PAL", "random": "Random"}
for name in names:
    if name not in aliases:
        aliases[name] = name

for i, path in enumerate(names):
    pal = pickle.load(open("../data/out/best_%s.dat" % path, 'r'))
    dist = [list(p).index(p[-1]) for p in pal]

    print min(dist), max(dist), path, len(dist)

    ax = plt.subplot(len(names), 1, i + 1)

    # the bins should be of integer width, because poisson is an integer distribution
    entries, bin_edges, _ = ax.hist(dist, label=aliases[path], bins=25, range=[-0.5, 240.5], normed=True)

    plt.legend(loc="center left", bbox_to_anchor=[1.0, 0.5], shadow=True, fancybox=True)
    plt.ylim(0, 0.055)

plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
fig.text(0.2, 0.95, "Histogram of Iterations to Global Convergence", fontsize=18)
fig.text(0.5, 0.02, 'Number of Iterations', ha='center', fontsize=16)
fig.text(0.02, 0.5, 'Normalized Probability', va='center', rotation='vertical', fontsize=16)

# plt.show()
plt.savefig("Hist_new.png")
