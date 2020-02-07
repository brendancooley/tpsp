import numpy as np
import os
import imp
import timeit
import time
import csv
import sys

import economy
import policies

import helpers_tpsp as hp

helpersPath = os.path.expanduser("~/Dropbox (Princeton)/14_Software/python/")
sys.path.insert(1, helpersPath)

import helpers
imp.reload(helpers)

mini = True
large = False

runEstimates = True

# dataFiles = os.listdir("tpsp_data/")

basePath = os.path.expanduser('~')
projectPath = basePath + "/Dropbox (Princeton)/1_Papers/tpsp/01_data/"

if mini is True:
    dataPath = projectPath + "tpsp_data_mini/"
    resultsPath = projectPath + "results_mini/"
elif large is True:
    dataPath = projectPath + "tpsp_data_large/"
    resultsPath = projectPath + "results_large/"
else:
    dataPath = projectPath + "tpsp_data/"
    resultsPath = projectPath + "results/"
helpers.mkdir(resultsPath)

estimatesPath = resultsPath + "estimates/"
counterfactualsPath = resultsPath + "counterfactuals/"
helpers.mkdir(estimatesPath)
helpers.mkdir(counterfactualsPath)

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
W = dists

N = len(Y)

E = Eq + Ex

data = {"tau":tau,"Xcif":Xcif,"Y":Y,"E":E,"r":r,"D":D,"W":W,"M":M, "ccodes":ccodes}  # Note: log distance

imp.reload(policies)
imp.reload(economy)
pecmy = policies.policies(data, params, ROWname, resultsPath)

theta_dict_init = dict()
theta_dict_init["c_hat"] = .1
theta_dict_init["alpha"] = .0001
theta_dict_init["gamma"] = 1.

theta_x_sv = pecmy.unwrap_theta(theta_dict_init)
_x, obj, status = pecmy.estimator(np.ones(pecmy.N), theta_x_sv, nash_eq=False)

print(_x)
print(obj)
print(status)

np.savetxt(estimatesPath + "x.csv", _x, delimiter=",")
