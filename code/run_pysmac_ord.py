'''

pySMAC 0.9.1 Installation:

URL: https://github.com/automl/pysmac
INSTALLATION: pip install git+https://github.com/automl/pysmac.git --user
'''

import os
import sys
import copy
import types
import random
import pysmac
import itertools
import subprocess
import scipy.stats
import numpy as np
import portable_pal as pal

if sys.version_info[0] == 2:
    import cPickle as pickle
elif sys.version_info[0] == 3:
    import _pickle as pickle
else:
    raise Exception("Error - Unknown python version!")


def f_min_obj_2(x0, x1, x2):
    return f_min_obj([x0, x1, x2])


def f_min_obj(x_int):
    # The actual function we are optimizing.  As we are minimizing,
    # and all data is actually the negated be, we need to return
    # the negative of the actually stored value
    data_x, data_y, combos, solvent_dict = f_min_obj.func_data
    index = f_min_obj.index

    x2 = solvent_dict[x_int[2]]

    sample = combos[x_int[0]] + [int(x_int[1]), int(x2)]

    for i, d in enumerate(data_x):
        if all([x == y for x, y in zip(sample, d)]):
            v = 1.0 * data_y[i]
            os.system("echo %f >> tmp_%d.dat" % (v, index))
            return -1.0 * v

    raise Exception("NOT FOUND")


def run_replication(index, x0, x1, x2, N_POINTS, func_data):
    # Setup and run our replication
    parameters = {
        "x0": ("categorical", x0, random.randint(0, 9)),
        "x1": ("categorical", x1, random.randint(0, 2)),
        "x2": ("ordinal", x2, random.choice(x2))
    }
    if os.path.exists("tmp_%d.dat" % index):
        os.remove("tmp_%d.dat" % index)
    opt = pysmac.SMAC_optimizer()
    f_min_obj.func_data = func_data
    f_min_obj.index = index

    xmin, fval = opt.minimize(f_min_obj_2, N_POINTS, parameters)

    best_y = [float(s) for s in open("tmp_%d.dat" % index).read().strip().split('\n')]

    os.remove("tmp_%d.dat" % index)
    # Assuming there is a possibility of converging sooner than all datapoints,
    # we will pad the ending of best_y with the best observed so far
    if len(best_y) != N_POINTS:
        best_y = best_y + [max(best_y) for _ in range(N_POINTS - len(best_y))]

    return best_y


def submit_job(run):
    '''
    This code will submit a job to the NBS queueing system so we can run our
    benchmarks in parallel!
    '''
    # Python script to run our replication
    py_script = '''
from run_pysmac_ord import run_replication
import cPickle as pickle

data = pickle.load(open("queue_helper/$INDEX_pysmac.pickle", 'rb'))
best_y = run_replication(*data)

pickle.dump(best_y, open("queue_helper/$INDEX_rep.out", 'wb'))
'''
    fptr = open("%d_rep.py" % run, 'w')
    fptr.write(py_script.replace("$INDEX", str(run)))
    fptr.close()

    # Submission script for this job (NBS)
    sub_script = '''
##NBS-name: "$INDEX_rep"
##NBS-nproc: 1
##NBS-queue: "bigmem"

source /fs/home/$USER/.zshrc

/fs/home/hch54/anaconda/bin/python2.7 -u $CUR_DIR/$INDEX_rep.py > $CUR_DIR/$INDEX_rep.log 2>&1
'''
    while "$INDEX" in sub_script:
        sub_script = sub_script.replace("$INDEX", str(run))
    while "$CUR_DIR" in sub_script:
        sub_script = sub_script.replace("$CUR_DIR", os.getcwd())
    name = "%d_rep" % run
    fptr = open("%s.nbs" % name, 'w')
    fptr.write(sub_script)
    fptr.close()

    # Submit job
    job_pipe = subprocess.Popen('jsub %s.nbs' % name, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    job_err = job_pipe.stderr.read()

    if "+notunique:" in job_err:
        raise Exception("Job with name %s already exists in the queue!" % name)

    job_id = job_pipe.stdout.read()
    job_id = job_id.split("submitted to queue")[0].split()[-1][2:-1]
    running_job = pal.Job(name, job_id=job_id)

    # Here we define a function to attach to the job class.
    def get_best_y(self):
        if not self.is_finished():
            self.wait()

        return pickle.load(open("queue_helper/%s.out" % self.name, 'rb'))

    # Attach the function
    running_job.get_best_y = types.MethodType(get_best_y, running_job)

    return running_job


def run_pysmac_ord(dataset, stats, NUM_RUNS=1000, on_queue=False):
    # -------------------------------------------------------------------------------------------------

    # Off-the-shelf BO (EI with GP model, no statistical model though)
    # http://www.jmlr.org/papers/volume16/neumann15a/neumann15a.pdf

    # Read in all the data points
    data = pickle.load(open("all_data.pickle", 'rb'))
    N_POINTS = len(data)

    data_x = [
        [row.index(1), row[3:].index(1), row[6:].index(1), row[9:].index(1), row[-2]]
        for row in data
    ]
    data_y = [row[-1] for row in data]
    combos = sorted([list(x) for x in list(set([tuple(sorted(v)) for v in itertools.product(range(3), repeat=3)]))])

    solvent_dict = {
        46.7: 0,
        36.7: 1,
        32.3: 2,
        40.24: 3,
        20.7: 4,
        10.9: 5,
        42.84: 6,
        35.9: 7
    }

    func_data = [data_x, data_y, combos, solvent_dict]

    # Ensure folders are as they should be
    if on_queue:
        if not os.path.isdir("queue_helper"):
            os.mkdir("queue_helper")
        else:
            os.system("rm queue_helper/*")

    # -------------------------------------------------------------------------------------------------

    each_run = []

    # Initialize our variable ranges
    x0 = list(range(10))
    x1 = list(range(3))
    x2 = sorted(solvent_dict.keys())

    # Run all replications
    jobs_on_queue = []
    for i in range(NUM_RUNS):
        dump_obj = [i, x0, x1, x2, N_POINTS, func_data]
        if not on_queue:
            pal.printProgressBar(i, NUM_RUNS, prefix="Running pySMAC with ordinal variables case...")
            best_y = run_replication(*dump_obj)
            each_run.append(best_y)
        else:
            pickle.dump(dump_obj, open("queue_helper/%d_pysmac.pickle" % i, 'wb'))
            jobs_on_queue.append(submit_job(i))

    if on_queue:
        for run, j in enumerate(jobs_on_queue):
            pal.printProgressBar(run, NUM_RUNS, prefix="Running pySMAC with ordinal variables case...")

            best_y = j.get_best_y()
            each_run.append(copy.deepcopy(best_y))

            # Delete the old files
            os.system("rm %d_rep.*" % run)

    # Parse the output data
    all_runs = [
        [max(r[:i + 1]) if i > 0 else x for i, x in enumerate(r)]
        for r in each_run
    ]

    n_to_max = []
    for this_run in all_runs:
        n_to_max.append(this_run.index(max(this_run)))

    each_run = np.array(all_runs)

    dataset["SMAC_ORD"] = [
        np.array([np.mean(each_run[:, i]) for i in range(N_POINTS)]),
        np.array([scipy.stats.sem(each_run[:, i]) for i in range(N_POINTS)])
    ]

    stats["SMAC_ORD"] = [n_to_max, N_POINTS]

    # Save all output
    pickle.dump(dataset["SMAC_ORD"], open("out/d_smac_ord.dat", 'wb'))
    pickle.dump(each_run, open("out/best_smac_ord.dat", 'wb'))

    pal.printProgressBar(NUM_RUNS, NUM_RUNS, prefix="Running pySMAC with ordinal variables case...", suffix="Done")
    return dataset, stats


if __name__ == "__main__":
    print("Running pySMAC with ordinal variables for only one replication! If you wish to do a full run, please use the run.py code.")
    dataset, stats = run_pysmac_ord({}, {}, NUM_RUNS=1, on_queue=False)
