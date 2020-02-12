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

location = sys.argv[1]
location = "local"

basePath = os.path.expanduser('~')

if location == "local":
    projectPath = basePath + "/Github/tpsp/"
if location == "hpc":
    projectPath = basePath + "/tpsp/"

helpersPath = os.path.expanduser(projectPath + "source/")
sys.path.insert(1, helpersPath)

import helpers

mini = False
large = True

runEstimates = True
computeCounterfactuals = False

if location == "local":
    dataAllPath = basePath + "/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
    if mini == True:
        dataPath = dataAllPath + "tpsp_data_mini/"
        resultsPath = dataAllPath + "results_mini/"
    if large == True:
        dataPath = dataAllPath + "tpsp_data_large/"
        resultsPath = dataAllPath + "results_large/"
if location == "hpc":
    dataPath = projectPath + "data/"
    resultsPath = projectPath + "results/"

estimatesPath = resultsPath + "estimates/"
counterfactualsPath = resultsPath + "counterfactuals/"

# Economic Parameters
beta = np.genfromtxt(dataPath + 'beta.csv', delimiter=',')
theta = np.genfromtxt(dataPath + 'theta.csv', delimiter=',')
mu = np.genfromtxt(dataPath + 'mu.csv', delimiter=',')
nu = np.genfromtxt(dataPath + 'nu.csv', delimiter=',')

params = {"beta":beta,"theta":theta,"mu":mu,"nu":nu}

# Data
tau = np.genfromtxt(dataPath + 'tau.csv', delimiter=',')
Xcif = np.genfromtxt(dataPath + 'Xcif.csv', delimiter=',')
Y = np.genfromtxt(dataPath + 'y.csv', delimiter=',')
Eq = np.genfromtxt(dataPath + 'Eq.csv', delimiter=',')
Ex = np.genfromtxt(dataPath + 'Ex.csv', delimiter=',')
r = np.genfromtxt(dataPath + 'r.csv', delimiter=',')
D = np.genfromtxt(dataPath + 'd.csv', delimiter=',')
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
# print(data)

### Estimate Model ###

if runEstimates == True:

    pecmy = policies.policies(data, params, ROWname, resultsPath)

    theta_dict_init = dict()
    theta_dict_init["c_hat"] = .1
    theta_dict_init["alpha"] = .0001
    theta_dict_init["gamma"] = 1.

    theta_x_sv = pecmy.unwrap_theta(theta_dict_init)
    xlvt_star, obj, status = pecmy.estimator(np.ones(pecmy.N), theta_x_sv, nash_eq=False)

    print(xlvt_star)
    print(obj)
    print(status)

    np.savetxt(estimatesPath + "x.csv", xlvt_star, delimiter=",")

### Load Estimates ###

if computeCounterfactuals == True:
    xlvt_star = np.genfromtxt(estimatesPath + 'x.csv', delimiter=',')
    xlvt_dict = pecmy.rewrap_xlvt(xlvt_star)
    theta_x_star = xlvt_dict["theta"]
    v_star = xlvt_dict["v"]

    x, obj, status = pecmy.estimator(v_star, theta_x_star, nash_eq=True)

    print(x)
    print(obj)
    print(status)

print("done.")
