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
import statsmodels.api as sm

import economy
import policies

import helpers_tpsp as hp
basePath = os.path.expanduser('~')

projectPath = basePath + "/Github/tpsp/"
projectFiles = basePath + "/Dropbox (Princeton)/1_Papers/tpsp/01_data/"

size = "mini/"

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

theta_dict_1 = dict()
theta_dict_1["c_hat"] = .25
theta_dict_1["alpha0"] = 0
theta_dict_1["alpha1"] = .001
theta_dict_1["gamma"] = .72

theta_dict_2 = dict()
theta_dict_2["c_hat"] = .78
theta_dict_2["alpha0"] = -5
theta_dict_2["alpha1"] = -.0001
theta_dict_2["gamma"] = 5

# TODO try just running inner loop, problem is that values of v change with theta as well, no reason we should run theta until covergence rather than iterating on v first.

imp.reload(policies)
pecmy = policies.policies(data, params, ROWname, results_path=resultsPath)  # generate pecmy and rcv vals

test = pecmy.alpha1_min(.01)

v_test = np.array([1.10, 1.37, 1.96, 1.11, 1.00, 1.19])
wv = pecmy.war_vals(v_test, pecmy.m, theta_dict_1)
np.diag(wv)
# wv = np.clip(wv, 0, np.inf)
id = 5
# np.fill_diagonal(wv, 0)

x_lbda, obj, status = pecmy.Lsolve_i_ipopt(id, v_test, wv[:,id])
np.savetxt("01_analysis/x.csv", x_lbda, delimiter=",")

x_lbda = np.genfromtxt("x.csv", delimiter=",")
ge_x = x_lbda[0:pecmy.x_len]
ge_dict1 = pecmy.ecmy.rewrap_ge_dict(ge_x)
ge_dict1["tau_hat"] * pecmy.ecmy.tau
wv[:,id]

pecmy.ecmy.geq_solve(ge_dict1["tau_hat"], np.ones(pecmy.N))

pecmy.G_hat(ge_x, v_test, id, all=True)
cons_test = pecmy.Lzeros_i_cons(x_lbda, np.zeros(pecmy.hhat_len + (pecmy.hhat_len + pecmy.N - 1) + pecmy.N + pecmy.N), id, v_test, wv[:,id])
cons_test[pecmy.hhat_len:]
cons_test[0:pecmy.hhat_len]

# x, obj, status = pecmy.estimator(v_test, pecmy.unwrap_theta(theta_dict_1), pecmy.mzeros, nash_eq=True)
