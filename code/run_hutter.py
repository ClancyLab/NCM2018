import os
import sys
import copy
import time
import types
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


def run_replication(run, START, use_eps, use_rho, data, data_x2, data_x, data_y, data_sp, NUM_REOPT, parallel):
    method = 'hutter'

    samples = pal.getSamples(START, eps=use_eps, rho=use_rho)
    samples = [pal.get_index(data_x2, np.array(s[:-1]), use_eps=use_eps, use_rho=use_rho) for s in samples]
    x = np.array([data_x[i] for i in samples])
    y = np.array([data_y[i] for i in samples])
    sp = np.array([data_sp[i] for i in samples])

    best_y = [max(y[:k]) if k > 0 else y[0] for k in range(START)]
    for j in range(len(data_x) - START):
        if (not j % NUM_REOPT) or (len(x) - NUM_REOPT < 0):
            # Update the hyperparameters
            if parallel:
                hps = pal.MLE_parallel(x, y, sp, method=method)
            else:
                hps = pal.MLE(x, y, sp, method=method)

            # Set prior
            mu = np.array([hps[-1] for i in data_x])

            if method == "simple":
                cov = pal.mk52(np.array([list(xx) + list(ss) for xx, ss in zip(data_x, data_sp)]), hps[:-2], hps[-2])
            elif method == "hutter":
                weights = [hps[0].tolist()] * 3 + [hps[1].tolist()] * 3 + [hps[2].tolist()] * 3 + [hps[3].tolist()] * 3 + hps[4:-2].tolist()
                cov = pal.mk52(np.array([list(xx) + list(ss) for xx, ss in zip(data_x, data_sp)]), weights, hps[-2])
            else:
                raise Exception("Method not accounted for!")

            # Update the posterior
            for sx, sy, ssp, sample in zip(x, y, sp, samples):
                mu, cov = pal.updatePosterior(copy.deepcopy(mu), copy.deepcopy(cov), sample, sy)

        # Predict best option via EI
        next_sample = pal.getNextSample(mu, cov, max(y), len(data), samples)
        assert next_sample not in samples, "Error - Sampled a point twice!"

        # Save Sample and Update the Posterior
        mu, cov = pal.updatePosterior(mu, cov, next_sample, data_y[next_sample])
        x = np.concatenate((x, copy.deepcopy([data_x[next_sample]])))
        y = np.concatenate((y, copy.deepcopy([data_y[next_sample]])))
        sp = np.concatenate((sp, copy.deepcopy([data_sp[next_sample]])))
        samples.append(next_sample)

        # Add to best_y
        best_y.append(max(y))

    return best_y


def submit_job(run):
    '''
    This code will submit a job to the NBS queueing system so we can run our
    benchmarks in parallel!
    '''
    # Python script to run our replication
    py_script = '''
from run_hutter import run_replication
import cPickle as pickle

data = pickle.load(open("queue_helper/$INDEX.pickle", 'rb'))
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
##NBS-queue: "short"

source /fs/home/$USER/.zshrc

export OMP_NUM_THREADS=1

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

    # Here we define a function to attach to the job class.  This allows
    # the user to call "x.mbo()" to retrieve the mbo when the job is done.
    def get_best_y(self):
        if not self.is_finished():
            self.wait()

        return pickle.load(open("queue_helper/%s.out" % self.name, 'rb'))

    # Attach the function
    running_job.get_best_y = types.MethodType(get_best_y, running_job)

    return running_job


def run_hutter(dataset, stats, NUM_RUNS=1000, parallel=True, on_queue=False):
    START = 1
    NUM_REOPT = 10

    # NOTE!!!! FOR SIMPLE AND HUTTER METHODS, WE ONLY HAVE PORTABLE PAL WRITTEN TO
    # ACCOUNT FOR use_rho = False and use_eps = True, SO DON"T CHANGE THIS!
    use_rho = False
    use_eps = True

    if not on_queue:
        if parallel and 'OMP_NUM_THREADS' not in os.environ:
            raise Exception("Trying to run in parallel, but OMP_NUM_THREADS is not defined.  Please export OMP_NUM_THREADS=1 before parallel run.")

    # Read in the data points
    data = pickle.load(open("all_data.pickle", 'rb'))
    data_x = np.array([d[:-4] for d in data])
    data_x2 = np.array([d[:-2] for d in data])
    data_y = np.array([d[-1] for d in data])
    data_sp = np.array([d[-4:-2] for d in data])

    # Error handle
    if not use_rho and not use_eps:
        raise Exception("Error - Either eps or rho should be specified.")

    if not use_rho:
        data_sp = np.array([[d[1]] for d in data_sp])
    if not use_eps:
        data_sp = np.array([[d[0]] for d in data_sp])

    N_POINTS = len(data_x)

    # Get samples from completed data
    each_run = []
    n_to_max = []

    # Ensure folders are as they should be
    if on_queue:
        if not os.path.isdir("queue_helper"):
            os.mkdir("queue_helper")
        else:
            os.system("rm queue_helper/*")

    # Run each replication
    jobs_on_queue = []
    for run in range(NUM_RUNS):
        if not on_queue:
            pal.printProgressBar(run, NUM_RUNS, prefix="Running HUTTER case...")

        dump_obj = [run, START, use_eps, use_rho, data, data_x2, data_x, data_y, data_sp, NUM_REOPT, parallel]
        if on_queue:
            pickle.dump(dump_obj, open("queue_helper/%d.pickle" % run, 'wb'))
            jobs_on_queue.append(submit_job(run))
        else:
            best_y = run_replication(*dump_obj)

            each_run.append(copy.deepcopy(best_y))
            n_to_max.append(best_y.index(max(best_y)))

    if on_queue:
        for run, j in enumerate(jobs_on_queue):
            pal.printProgressBar(run, NUM_RUNS, prefix="Running HUTTER case...")

            best_y = j.get_best_y()

            each_run.append(copy.deepcopy(best_y))
            n_to_max.append(best_y.index(max(best_y)))

            # Delete the old files
            os.system("rm %d_rep.*" % run)

    # Parse and save data
    each_run = np.array(each_run)
    dataset["HUTTER"] = [
        np.array([np.mean(each_run[:, i]) for i in range(N_POINTS)]),
        np.array([scipy.stats.sem(each_run[:, i]) for i in range(N_POINTS)])
    ]

    pickle.dump(dataset["HUTTER"], open("out/d_hutter.dat", 'wb'))
    pickle.dump(each_run, open("out/best_hutter.dat", 'wb'))

    stats["HUTTER"] = [n_to_max, N_POINTS]

    pal.printProgressBar(NUM_RUNS, NUM_RUNS, prefix="Running HUTTER case...", suffix="Done")
    return dataset, stats


if __name__ == "__main__":
    print("Running HUTTER for only one replication! If you wish to do a full run, please use the run.py code.")
    t0 = time.time()
    dataset, stats = run_hutter({}, {}, NUM_RUNS=2, parallel=True, on_queue=True)
