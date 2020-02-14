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

location = sys.argv[1]  # local, hpc
size = sys.argv[2] # mini/, mid/, large/
# location = "local"
# size = "mini/"

basePath = os.path.expanduser('~')

if location == "local":
    projectPath = basePath + "/Github/tpsp/"
if location == "hpc":
    projectPath = basePath + "/tpsp/"

if location == "local":
    projectFiles = basePath + "/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
if location == "hpc":
    projectFiles = projectPath

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

helpers.mkdir(resultsPath)
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

### Estimate Model ###

pecmy = policies.policies(data, params, ROWname, resultsPath)

if runEstimates == True:

    theta_dict_init = dict()
    theta_dict_init["c_hat"] = .1
    theta_dict_init["alpha"] = .0001
    theta_dict_init["gamma"] = 1.

    theta_x_sv = pecmy.unwrap_theta(theta_dict_init)

    start_time = time.time()
    xlvt_star, obj, status = pecmy.estimator(np.ones(pecmy.N), theta_x_sv, nash_eq=False)
    print("--- Estimator converged in %s seconds ---" % (time.time() - start_time))

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

# test hpc
# if location == "hpc":
#     test_out = np.ones(len(M))
#     np.savetxt(estimatesPath + "test.csv", test_out, delimiter=",")

print("done.")
