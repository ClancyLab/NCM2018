# The Physical Analytics pipeLine (PAL) - Code Distribution

## What is PAL?
PAL is a collaborative approach to Materials Science, in which Bayesian Optimization, Computational Chemistry, and Experiments can come together to rapidly seek out an optimal molecular configuration.  In this case, it is applied to the study of Hybrid Organic-Inorganic Perovskites (HOIPs).

## Implementation
HOIPs are exciting materials of interest due to their being solution processable solar cells.  Unfortunately, the configurational space for which HOIPs exists is exceedingly large for standard, brute-force approaches (both experimental and computational).  As such, we (1) reduce the focus to so me computational objective function that can then be (2) optimized *via* Bayesian optimization.  Experimentally, HOIPs are fabricated (in a simplistic sense) by mixing salts in solution, and allowing for nucleation when spin-coating on a surface.  During this process, many different things form, such as other salts or polycrystalline forms.  As such, a simple starting point for observation is to focus on the solvation of HOIP salts.  That is, how well are the salts solvated?  This leads to the computational objective function.

### Computational Objective Function - Binding Energy
An ideal objective to define solvation is that of the [enthalpy of solvation](https://en.wikipedia.org/wiki/Enthalpy_change_of_solution).  Unfortunately, this is a fairly expensive calculation, as it requires a full solvation shell of solvents.  Further, with there being no multi-purpose perovskite/solvent force field (as of April 9th, 2018) that allows for the study of any arbitrary perovskite-solvent mixture (in Molecular Dynamics), we restrict calculations to Density Functional Theory (DFT).  We continue with the smallest system: the intermolecular binding energy between one solvent molecule and some perovskite salt (ABX<sub>3</sub>).  This data is pre-calculated (using the [Orca DFT software](https://orcaforum.cec.mpg.de/)) and held within a python2 cPickle file (data/all_data.pickle).

### Bayesian Optimization
The Bayesian optimization is done by our own custom code.  It works as follows:

1. Take a sample (X) of the combinatorial space *via* [Latin Hypercube Sampling (LHS)](https://pythonhosted.org/pyDOE/randomized.html) (accomplished using the [pyDOE package](https://github.com/tisimst/pyDOE/blob/master/pyDOE/doe_lhs.py)).
2. Obtain values from our objective function: Y(X).
3. Using our custom prior (see paper), the [Maximum Likelihood Estimation (MLE) of a normal distribution](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation#Continuous_distribution,_continuous_parameter_space), a few LHS sampled hyperparameters, and the [scipy L-BFGS-B optimizer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize) we find an ideal set of hyperparameters.
4. Update the posterior with all the sampled points up to now.
5. Using the [Expected Improvement (EI)](https://arxiv.org/pdf/1506.01349.pdf), choose a combination x\* to sample: y\* = Y(x\*)
6. Update the posterior.  If we want to recalculate hyperparameters (maybe every 10 iterations), go back to step 3, else go back to step 5.  Repeat until no available samples remain.

### Data Structures

One can read in the data from the paper quite easily as follows (note, this is python2.7 and the code is in the code/test.py file):

```python
import cPickle as pickle

# Note - to import this you'll either have to be in the same folder as plotter
# and portable_pal, or add that path to your PYTHONPATH environment variable.
from plotter import plot
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
stats, _ = pickle.load(open("../data/final.pickle", 'r'))

# Plot it if you want
plot(stats)
```

### Generating Plot from Paper

If you wish to generate the plot from the paper, the code example in Data Structures will do just that.  However, if you want to re-run the optimization yourself, you can do so by simply navigating to the code folder and running the **run.py** script.  Note, this has been re-written slightly to allow for running in both python2.7 and python3.  If run in python2.7, you may see timeouts during the pySMAC simulation (results remain the same).  If running in python3, these timeouts will no longer appear.  All output is generated in an **out** subdirectory, such as the plot itself.  Finally, a table similar to that in the paper will be automatically calculated and printed to the terminal at the end of the calculations.  We recommend reducing the number of replications from 1000 to 100 for pySMAC and PAL for speed.
