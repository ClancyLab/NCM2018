'''
This file should be all inclusive for moving around in regards to benchmark code.
'''
import re
import sys
import time
import scipy
from pyDOE import doe_lhs
import subprocess
import getpass
import numpy as np
import scipy.linalg
import multiprocessing as mp
from scipy.stats import norm
from scipy import optimize as op
from scipy.spatial import distance_matrix

# Information on solvents
solvents = {
    "DMSO": {"density": 1.0, "dielectric": 46.7, "index": 0},
    "DMF": {"density": 0.95, "dielectric": 36.7, "index": 1},
    "NMP": {"density": 1.1, "dielectric": 32.3, "index": 2},
    "GBL": {"density": 1.1, "dielectric": 40.24, "index": 3},
    "ACE": {"density": 0.78, "dielectric": 20.7, "index": 4},  # ACETONE
    "MCR": {"density": 0.85, "dielectric": 10.9, "index": 5},  # METHACROLEIN
    "THTO": {"density": 1.2, "dielectric": 42.84, "index": 6},
    "NM": {"density": 1.14, "dielectric": 35.9, "index": 7},  # NITROMETHANE
}
solvent_names = [k for k, _ in solvents.items()]

# Default error to be used
error = lambda: 0.0
# Number of processors to use
PROCESSES_ALLOWED = 4
# Number of random starting parameters in MLE optimization
NUMBER_OF_RANDOM_STARTING_PARAMETERS = 8


def parseNum(num):
    '''
    Convert descriptor into a name.  Note, this is extensible to if the IS exists, or if it doesn't
    (determined by num_has_IS).  If the IS does exist, it MUST be the first index of num.

    **Parameters**

    **Returns**

        name: *str*
            String representation of this system.  This looks like:
                cation + "Pb" + H1 + H2 + H3 + "_" + solvent + "_" + IS
            If num_has_IS = False, then ("_" + IS) is left out
    '''

    HALIDE_STRS = ["Br", "Cl", "I"]
    CATION_STRS = ["MA", "FA", "Cs"]
    solvents = {
        "DMSO": {"name": "DMSO", "density": 1.0, "dielectric": 46.7, "index": 0},
        "DMF": {"name": "DMF", "density": 0.95, "dielectric": 36.7, "index": 1},
        "NMP": {"name": "NMP", "density": 1.1, "dielectric": 32.3, "index": 2},
        "GBL": {"name": "GBL", "density": 1.1, "dielectric": 40.24, "index": 3},
        "ACE": {"name": "ACE", "density": 0.78, "dielectric": 20.7, "index": 4},  # ACETONE
        "MCR": {"name": "MCR", "density": 0.85, "dielectric": 10.9, "index": 5},  # METHACROLEIN
        "THTO": {"name": "THTO", "density": 1.2, "dielectric": 42.84, "index": 6},
        "NM": {"name": "NM", "density": 1.14, "dielectric": 35.9, "index": 7},  # NITROMETHANE
    }

    offset = 0

    h1 = HALIDE_STRS[num[offset:].index(1)]
    h2 = HALIDE_STRS[num[3 + offset:].index(1)]
    h3 = HALIDE_STRS[num[6 + offset:].index(1)]
    offset += 9

    c = CATION_STRS[num[offset:].index(1)]

    s = [v["name"] for k, v in solvents.items() if v["index"] == num[-1]][0]

    h1, h2, h3 = sorted([h1, h2, h3])

    return "%sPb%s%s%s_%s" % (c, h1, h2, h3, s)


def get_index(me, obj, use_eps=True, use_rho=True):
    '''
    Will find the index of a list in a list.
    '''
    # In the case that we use_eps but not use_rho, we can flip the
    # last two columns of me so we check correctly
    if use_eps and not use_rho:
        me = np.array(me)
        me[:, [-2, -1]] = me[:, [-1, -2]]
    for i, row in enumerate(me):
        if all([a == b for a, b in zip(row, obj)]):
            return i
    raise Exception("obj %s was not found in list." % str(obj))


def updatePosterior(mu, cov, x, y, debug=None):
    '''
    Given a new sampled point, update the posterior
    '''

    assert cov[x, x] != 0, "Error - Failure in Posterior Update in which variance of sampled point is too low!"

    cov_vec = cov[x, :]
    mu_new = mu + (y - mu[x]) / (cov[x, x] + error()) * cov_vec
    cov_new = cov - np.outer(cov_vec, cov_vec) / (cov[x, x] + error())

    return mu_new, cov_new


def opt_hps_parallel(dat):
    x, y, sp, val, bnds = dat
    f = lambda *args: -1.0 * likelihood(x, y, sp, None, *args)
    res = op.minimize(f, val, bounds=bnds)
    return res['x'], res.fun


def opt_hps_parallel_simple(dat):
    x, y, sp, val, bnds = dat
    f = lambda *args: -1.0 * likelihood(x, y, sp, 'simple', *args)
    res = op.minimize(f, val, bounds=bnds)
    return res['x'], res.fun


def opt_hps_parallel_hutter(dat):
    x, y, sp, val, bnds = dat
    f = lambda *args: -1.0 * likelihood(x, y, sp, 'hutter', *args)
    res = op.minimize(f, val, bounds=bnds)
    return res['x'], res.fun


def MLE_parallel(x, y, sp, n_start=None, method=None):
    '''
    Given sampled data, use the maximum likelihood-estimator to find
    hyperparameters.

    **Returns**

        hyperparams: *list, float/int*
    '''

    # ASSUME: HP = [mu_alpha, sig_alpha, sig_beta, mu_zeta, sig_zeta, sig_m, l1, l2]

    import numpy as np

    if n_start is None:
        n_start = NUMBER_OF_RANDOM_STARTING_PARAMETERS  # How many samples we do for MLE
    bounds = [
        (1E-3, max(y)),
        (1E-3, np.var(y)),
        (1E-3, np.var(y)),
        (1E-3, max(y)),
        (1E-3, np.var(y)),
        (1E-3, np.var(y)),
        (1E-3, 1),
        (1E-3, 1)
    ]

    if len(sp[0]) == 1:
        bounds = bounds[:-2] + [(1.0, 1.0)]

    if method is not None and method == "simple":
        bounds = [
            (1E-3, 1),  # h1_Br
            (1E-3, 1),  # h1_Cl
            (1E-3, 1),  # h1_I
            (1E-3, 1),  # h2_Br
            (1E-3, 1),  # h2_Cl
            (1E-3, 1),  # h2_I
            (1E-3, 1),  # h3_Br
            (1E-3, 1),  # h3_Cl
            (1E-3, 1),  # h3_I
            (1E-3, 1),  # c_Cs
            (1E-3, 1),  # c_FA
            (1E-3, 1),  # c_MA
            (1E-3, 1),  # Dielectric
            (1E-3, 1),  # sigma
            (1E-3, max(1.0, max(y)))  # constant prior
        ]
    elif method is not None and method == "hutter":
        bounds = [
            (1E-3, 1),  # h1
            (1E-3, 1),  # h2
            (1E-3, 1),  # h3
            (1E-3, 1),  # c
            (1E-3, 1),  # Dielectric
            (1E-3, 1),  # sigma
            (1E-3, max(1.0, max(y)))  # constant prior
        ]

    sampled_values = doe_lhs.lhs(len(bounds), samples=n_start)

    init_values = [
        (x, y, sp, [s * (b[1] - b[0]) + b[0] for s, b in zip(sampled_values[j], bounds)], bounds)
        for j in range(n_start)
    ]

    mle_list = np.zeros([n_start, len(bounds)])
    lkh_list = np.zeros(n_start)

    pool = mp.Pool(processes=PROCESSES_ALLOWED)

    if method is not None and method == "simple":
        all_res = pool.map(opt_hps_parallel_simple, init_values)
    elif method is not None and method == "hutter":
        all_res = pool.map(opt_hps_parallel_hutter, init_values)
    else:
        all_res = pool.map(opt_hps_parallel, init_values)
    pool.terminate()

    for i, res in zip(range(n_start), all_res):
        mle_list[i, :], lkh_list[i] = res

    # Now, select parameters for the max likelihood of these
    index = np.nanargmin(lkh_list)  # Note, min because we inverted the likelihood so we can use a minimizer.
    best_theta = mle_list[index, :]

    return best_theta


def MLE(x, y, sp, n_start=None, method=None):
    '''
    Given sampled data, use the maximum likelihood-estimator to find
    hyperparameters.

    **Returns**

        hyperparams: *list, float/int*
    '''

    # ASSUME: HP = [mu_alpha, sig_alpha, sig_beta, mu_zeta, sig_zeta, sig_m, l1, l2]

    if n_start is None:
        n_start = NUMBER_OF_RANDOM_STARTING_PARAMETERS  # How many samples we do for MLE
    bounds = [
        (1E-3, max(y)),
        (1E-3, np.var(y)),
        (1E-3, np.var(y)),
        (1E-3, max(y)),
        (1E-3, np.var(y)),
        (1E-3, np.var(y)),
        (1E-3, 1),
        (1E-3, 1)
    ]

    if len(sp[0]) == 1:
        bounds = bounds[:-2] + [(1.0, 1.0)]

    if method is not None and method == 'simple':
        bounds = [
            (1E-3, 1),  # h1_Br
            (1E-3, 1),  # h1_Cl
            (1E-3, 1),  # h1_I
            (1E-3, 1),  # h2_Br
            (1E-3, 1),  # h2_Cl
            (1E-3, 1),  # h2_I
            (1E-3, 1),  # h3_Br
            (1E-3, 1),  # h3_Cl
            (1E-3, 1),  # h3_I
            (1E-3, 1),  # c_Cs
            (1E-3, 1),  # c_FA
            (1E-3, 1),  # c_MA
            (1E-3, 1),  # Dielectric
            (1E-3, 1),  # sigma
            (1E-3, max(y))  # constant prior
        ]
    elif method is not None and method == "hutter":
        bounds = [
            (1E-3, 1),  # h1
            (1E-3, 1),  # h2
            (1E-3, 1),  # h3
            (1E-3, 1),  # c
            (1E-3, 1),  # Dielectric
            (1E-3, 1),  # sigma
            (1E-3, max(y))  # constant prior
        ]

    sampled_values = doe_lhs.lhs(len(bounds), samples=n_start)

    init_values = [
        [s * (b[1] - b[0]) + b[0] for s, b in zip(sampled_values[j], bounds)]
        for j in range(n_start)
    ]

    mle_list = np.zeros([n_start, len(bounds)])
    lkh_list = np.zeros(n_start)
    # MLE = Maximum Likelihood Estimation.  But we use a minimizer! So invert the
    # likelihood instead.
    f = lambda *args: -1.0 * likelihood(x, y, sp, method, *args)

    # For each possible starting of parameters, minimize and store the resulting likelihood
    for i in range(n_start):
        results = op.minimize(f, init_values[i], bounds=bounds)
        mle_list[i, :] = results['x']  # Store the optimized parameters
        lkh_list[i] = results.fun  # Store the resulting likelihood

    # Now, select parameters for the max likelihood of these
    index = np.nanargmin(lkh_list)  # Note, min because we inverted the likelihood so we can use a minimizer.
    best_theta = mle_list[index, :]

    return best_theta


def likelihood(x, y, sp, method, theta):
    '''
    This function computes the likehood of solubilities given hyper parameters
    in the list theta.

    **Parameters**

        X:
            A list of the sampled X coordinates.
        Y:
            A list of the objectives calculated, corresponding to the different X values.
        S:
            A list of lists, holding the solvent properties.
        mean:
            Function that, given X, Y, S, and theta, will calculate the mean vector.
        cov:
            Function that, given X, Y, S, and theta, will calculate the covariance matrix.
        theta:
            Object holding hyperparameters

    **Returns**

        likelihood: *float*
            The log of the likelihood without the constant term.
    '''

    # ASSUME: HP = [mu_alpha, sig_alpha, sig_beta, mu_zeta, sig_zeta, sig_m, l1, l2]

    if method is None:
        simple, hutter = False, False
    else:
        simple = method == "simple"
        hutter = method == "hutter"

    if simple:
        mu = np.array([theta[-1] for i in x])
        cov = mk52([list(xx) + list(ss) for xx, ss in zip(x, sp)], theta[:-2], theta[-2])
    elif hutter:
        mu = np.array([theta[-1] for i in x])
        weights = [theta[0].tolist()] * 3 + [theta[1].tolist()] * 3 + [theta[2].tolist()] * 3 + [theta[3].tolist()] * 3 + theta[4:-2].tolist()
        cov = mk52([list(xx) + list(ss) for xx, ss in zip(x, sp)], weights, theta[-2])
    else:
        mu = np.array([4 * theta[0] + theta[3] for i in x])
        cov = theta[1] * np.dot(x, x.T) +\
            theta[2] * np.diag(np.ones(len(x))) +\
            theta[4] +\
            mk52(sp, theta[-len(sp[0]):], theta[-(len(sp[0]) + 1)])

    Y = np.array(y) - mu

    # Random perturbation for Sig so that we don't run into the issue of
    # having singular matrix during numerical optimization
    # rand_pert = np.random.random(cov.shape) * 1E-4
    rand_pert = 0

    # Current method of using cholesky decomp with scipy
    # L = scipy.linalg.cho_factor(cov, lower=True, overwrite_a=False)
    L = scipy.linalg.cho_factor(cov + rand_pert, lower=True, overwrite_a=False)
    alpha = scipy.linalg.cho_solve(L, Y)
    return -0.5 * Y.T.dot(alpha) - np.sum([np.log(v) for v in np.diag(L[0])]) - len(mu) / 2.0 * np.log(2.0 * np.pi)


def mk52(data, weights, sig):
    '''
    Compute the 5/2 matern kernel from a data set, some weights for the different
    data points, and a variance.

    **Parameters**

        data: *list, list, float*
            A list of lists, holding floats which are our data points.

        weights: *list, float*
            A list of weights for the data points

        sig: *float*
            A variance for our points

    **Returns**

        kernel: *list, list, float*
            An n-by-n matrix holding the kernel.  n is the number of data
            points in the input data object.
    '''

    # First step, let's ensure we have correct object
    if isinstance(weights, float):
        weights = [weights]
    # print weights 
    data, weights = np.array(data), np.array(weights)**0.5

    if np.isnan(weights[0]):
        raise Exception("Error - Issue in weights being NaN.")
    n, dim = data.shape

    # Get the pairwise distance matrix between data points.  Note, because we
    # want to weight the distances based on each dimension, we first multiply
    # each dimension by the sqrt(weight)
    pairwise_matrix = distance_matrix(data * weights, data * weights)

    # Now, we can get the kernel... not sure why sqrt(5)
    K = np.sqrt(5) * pairwise_matrix
    return sig * (1.0 + K + K**2 / 3.0) * np.exp(-K)


def getNextSample(mu, cov, best, dim, samples):
    '''
    This function determines the next
    '''
    EI_list = np.array(
        [(mu[i] - best) * norm.cdf((mu[i] - best) / np.sqrt(cov[i, i])) +
         np.sqrt(cov[i, i]) * norm.pdf((mu[i] - best) / np.sqrt(cov[i, i]))
         # if np.nanmax([cov[i, i], 0.0]) > 0 else 0
         if (cov[i, i] > 0 and i not in samples) else 0
         for i in range(dim)]).reshape((-1, dim))[0]
    next_sample = np.nanargmax(EI_list)

    if np.nanmax([EI_list[next_sample], 0]) <= 0:
        # print("Warning - EI list dropped to zeros. Will select one by random")
        next_sample = [i for i in range(dim) if i not in samples]
        return np.random.choice(next_sample)

    return next_sample


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=20, fill='+', buf=None):
    """
    NOTE! THIS IS COPIED FROM STACK OVERFLOW (with minor changes), USER Greenstick
    Link: https://stackoverflow.com/a/34325723

    Call in a loop to create terminal progress bar.

    **Parameters**

        iteration: *int*
            Current iteration.
        total: *int*
            Total number of iterations.
        prefix: *str, optional*
            Prefix for the loading bar.
        suffix: *str, optional*
            Suffix for the loading bar.
        decimals: *int, optional*
            Positive number of decimals in percent complete
        length: *int, optional*
            Character length of the loading bar.
        fill: *str, optional*
            Bar fill character.
    """
    if buf is not None:
        if not hasattr(printProgressBar, "buf"):
            setattr(printProgressBar, "buf", buf)
        elif printProgressBar.buf < 0:
            printProgressBar.buf = buf
        else:
            printProgressBar.buf -= 1
            return

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total:
        sys.stdout.write('\n')

    sys.stdout.flush()


def getSamples(N_samples, eps=True, rho=True):

    if not rho and not eps:
        raise Exception("Error - Either eps or rho should be specified.")

    samples = doe_lhs.lhs(5, N_samples)

    trio = lambda v: [int(v > (chk - 1.0 / 3.0) and v <= chk) for chk in [1. / 3., 2. / 3., 1.0]]
    solvent_ranges = [i * 1.0 / len(solvents) for i in range(1, len(solvents) + 1)]

    solv = lambda v: solvent_names[[v <= s for s in solvent_ranges].index(True)]

    samples = [
        trio(s[0]) + trio(s[1]) + trio(s[2]) + trio(s[3]) +
        [solvents[solv(s[-1])]["density"],
         solvents[solv(s[-1])]["dielectric"],
         solvents[solv(s[-1])]["index"]]
        for s in samples]

    if not rho:
        samples = [s[:-3] + s[-2:] for s in samples]
    if not eps:
        samples = [s[:-2] + [s[-1]] for s in samples]

    # order the first three trios by Br -> Cl -> I
    for i, sample in enumerate(samples):
        t0 = sample[0:3]
        t1 = sample[3:6]
        t2 = sample[6:9]
        ts = [t0, t1, t2]
        char = []
        for t in ts:
            char.append(["Br", "Cl", "I"][t.index(1)])
        dat = [(c, t) for c, t in zip(char, ts)]
        dat = sorted(dat, key=lambda x: x[0])
        samples[i] = dat[0][1] + dat[1][1] + dat[2][1] + sample[9:]

    # Ensure no duplicates
    samples = [tuple(s) for s in samples]
    samples = [list(s) for s in set(samples)]

    return samples


def log(dat):
    fptr = open("log.dat", 'a')
    fptr.write(str(dat) + "\n")
    fptr.close()


class Job(object):
    """
    Job class to wrap simulations for queue submission.

    **Parameters**

        name: *str*
            Name of the simulation on the queue.
        process_handle: *process_handle, optional*
            The process handle, returned by subprocess.Popen.

    **Returns**

        This :class:`Job` object.
    """
    def __init__(self, name, process_handle=None, job_id=None):
        self.name = name
        self.process_handle = process_handle
        self.job_id = job_id

    def wait(self, tsleep=10, verbose=False):
        """
        Hang until simulation has finished.

        **Returns**

            None
        """
        if self.process_handle is not None:
            self.process_handle.wait()
        else:
            while True:
                if not self.is_finished():
                    if verbose:
                        print("Job (%s) is still running..." % self.name)
                    time.sleep(tsleep)
                else:
                    break

    def is_finished(self):
        """
        Check if simulation has finished or not.

        **Returns**

            is_on_queue: *bool*
                Whether the simulation is still running (True), or not (False).
        """
        if self.process_handle is not None:
            return self.process_handle.poll() == 0
        if self.job_id is not None:
            running = any([self.job_id in j for j in get_all_jobs(detail=3)])
        else:
            running = self.name in get_all_jobs(detail=0)
        return not running


def get_all_jobs(queueing_system="nbs", detail=0):
    """
    Get a list of all jobs currently on your queue.  The *detail*
    variable can be used to specify how much information you want returned.

    **Parameters**

        queueing_system: *str, optional*
            Which queueing system you are using (NBS or PBS).
        detail: *int, optional*
            The amount of information you want returned.

    **Returns**

        all_jobs: *list*
            Depending on *detail*, you get the following:

                - *details* =0: *list, str*
                    List of all jobs on the queue.

                - *details* =1: *list, tuple, str*
                    List of all jobs on the queue as:
                        (job name, time run, job status)

                - *details* =2: *list, tuple, str*
                    List of all jobs on the queue as:
                        (job name,
                         time run,
                         job status,
                         queue,
                         number of processors)
    """
    if queueing_system.strip().lower() == "nbs":
        # Get input from jlist as a string
        p = subprocess.Popen(['jlist'], stdout=subprocess.PIPE)
        output = p.stdout.read()

        # Get data from string
        pattern = getpass.getuser() +\
            '''[\s]+([\S]+)[\s]+([\S]+)[\s]+([\S]+)'''
        info = re.findall(pattern, output)

        # Get a list of names
        names = []
        for a in info:
            names.append(a[0])

        if len(names) > 0:
            out_ids = output.split("\n")
            out_ids = [x.split()[0] for x in out_ids if len(x.split()) > 0 and _isFloat(x.split()[0])]
            info = [tuple(list(i) + [j]) for i, j in zip(info, out_ids)]

        # If user wants more information
        if detail == 3:
            return [i[-1] for i in info]
        if detail == 2:
            for i, a in enumerate(info):
                p = subprocess.Popen(['jshow', a[0]], stdout=subprocess.PIPE)
                s = p.stdout.read()
                serv = s[s.find('Queue name:'):].split()[2].strip()
                try:
                    threads = s[s.find('Slot Reservations'):].split()[4]
                    threads = threads.strip()
                except:
                    threads = 1
                info[i] = info[i] + (serv, threads,)
            return info

        # Return appropriate information
        if detail == 1:
            return info
        else:
            return names

    elif queueing_system.strip().lower() == "pbs":
        # Do This
        raise Exception("THIS CODE NOT WRITTEN YET.")
    elif queueing_system.strip().lower() == "slurm":
        p = subprocess.Popen(['showq'], stdout=subprocess.PIPE)
        output = p.stdout.read().split('\n')
        all_jobs = [job.strip().split()
                    for job in output if getpass.getuser() in job]
        if detail == 2:
            all_jobs = [(j[1], j[5], j[3], 'TACC', j[4]) for j in all_jobs]
        elif detail == 1:
            all_jobs = [(j[1], j[5], j[3]) for j in all_jobs]
        else:
            all_jobs = [j[1] for j in all_jobs]
        return all_jobs
    else:
        raise Exception("Unknown queueing system passed to get_all_jobs. \
Please choose NBS, PBS, or SLURM for now.")


def _isFloat(x):
    try:
        float(x)
    except (ValueError, TypeError):
        return False
    return True
