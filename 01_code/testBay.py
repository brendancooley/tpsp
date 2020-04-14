import autograd as ag
import autograd.numpy as np
import os
import imp
import timeit
import time
import csv
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
import statsmodels.api as sm

import economy
import policies

import helpers_tpsp as hp
basePath = os.path.expanduser('~')

projectPath = basePath + "/Github/tpsp/"
projectFiles = basePath + "/Dropbox (Princeton)/1_Papers/tpsp/01_data/"


size = "mid/"
sv_fname = "out/mini_sv.csv"
out_fname = "out/mid_test_mumps.csv"
# sv = np.genfromtxt(sv_fname, delimiter=',')

helpersPath = os.path.expanduser(projectPath + "source/")
sys.path.insert(1, helpersPath)

import helpers

runEstimates = True
computeCounterfactuals = False

data_dir_base = projectFiles + "data/"
results_dir_base = projectFiles + "results/"

dataPath = data_dir_base + size
resultsPath = results_dir_base + size

estimatesPath = resultsPath + "estimates/"
counterfactualsPath = resultsPath + "counterfactuals/"

estimatesPath = resultsPath + "estimates/"

# Economic Parameters
beta = np.genfromtxt(dataPath + 'beta.csv', delimiter=',')
theta = np.genfromtxt(dataPath + 'theta.csv', delimiter=',')
mu = np.genfromtxt(dataPath + 'mu.csv', delimiter=',')
nu = np.genfromtxt(dataPath + 'nu.csv', delimiter=',')

params = {"beta":beta,"theta":theta,"mu":mu,"nu":nu}

# Data
tau = np.genfromtxt(dataPath + 'tau.csv', delimiter=',')
Xcif = np.genfromtxt(dataPath + 'Xcif.csv', delimiter=',')
Y = np.genfromtxt(dataPath + 'Y.csv', delimiter=',')
Eq = np.genfromtxt(dataPath + 'Eq.csv', delimiter=',')
Ex = np.genfromtxt(dataPath + 'Ex.csv', delimiter=',')
r = np.genfromtxt(dataPath + 'r.csv', delimiter=',')
D = np.genfromtxt(dataPath + 'D.csv', delimiter=',')
ccodes = np.genfromtxt(dataPath + 'ccodes.csv', delimiter=',', dtype="str")
dists = np.genfromtxt(dataPath + 'cDists.csv', delimiter=',')
M = np.genfromtxt(dataPath + "milex.csv", delimiter=",")
ROWname = np.genfromtxt(dataPath + 'ROWname.csv', delimiter=',', dtype="str")
ROWname = str(ROWname)

M = M / np.min(M)  # normalize milex
# W = np.log(dists+1)
W = dists

N = len(Y)

E = Eq + Ex

data = {"tau":tau,"Xcif":Xcif,"Y":Y,"E":E,"r":r,"D":D,"W":W,"M":M, "ccodes":ccodes}  # Note: log distance

# v = np.ones(N)
# v = np.array([1.08, 1.65, 1.61, 1.05, 1.05, 1.30])
# v = np.repeat(1.4, N)

# TODO try just running inner loop, problem is that values of v change with theta as well, no reason we should run theta until covergence rather than iterating on v first.

imp.reload(policies)
imp.reload(economy)
pecmy = policies.policies(data, params, ROWname, results_path=resultsPath)

theta_dict = dict()
theta_dict["eta"] = 1.
theta_dict["c_hat"] = 25.
theta_dict["alpha1"] = 0.
theta_dict["alpha2"] = 0.
theta_dict["gamma"] = 0.
theta_dict["C"] = np.repeat(25., pecmy.N)
theta_x = pecmy.unwrap_theta(theta_dict)


# opt.root(pecmy.pp_wrap_alpha, .5, args=(.99, ))['x']
# pecmy.W ** - .75

v = np.mean(pecmy.ecmy.tau, axis=1)

# x, obj, status = pecmy.estimator(v, theta_x, pecmy.m, sv=sv, nash_eq=False)
x, obj, status = pecmy.estimator(v, theta_x, pecmy.m, sv=None, nash_eq=False)

x_dict = pecmy.rewrap_xlhvt(x)
ge_dict = pecmy.ecmy.rewrap_ge_dict(x_dict["ge_x"])
theta_dict = pecmy.rewrap_theta(x_dict["theta"])

print(ge_dict["tau_hat"]*pecmy.ecmy.tau)
print("-----")
for i in theta_dict.keys():
    print(i)
    print(theta_dict[i])


np.savetxt(out_fname, x, delimiter=",")
