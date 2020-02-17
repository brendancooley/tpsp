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
computeCounterfactuals = True

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
    theta_dict_init["c_hat"] = .2
    theta_dict_init["alpha0"] = 0
    theta_dict_init["alpha1"] = 0
    theta_dict_init["gamma"] = 1.

    theta_x_sv = pecmy.unwrap_theta(theta_dict_init)

    start_time = time.time()
    xlvt_star, obj, status = pecmy.estimator(np.repeat(1., pecmy.N), theta_x_sv, pecmy.m, nash_eq=False)
    print("--- Estimator converged in %s seconds ---" % (time.time() - start_time))

    print(xlvt_star)
    print(obj)
    print(status)

    xlvt_star_path = estimatesPath + "x.csv"
    np.savetxt(xlvt_star_path, xlvt_star, delimiter=",")

### Save Estimates ###

xlvt_star = np.genfromtxt(xlvt_star_path, delimiter=",")
ge_x_star = pecmy.rewrap_xlvt(xlvt_star)["ge_x"]
v_star = pecmy.rewrap_xlvt(xlvt_star)["v"]
theta_x_star = pecmy.rewrap_xlvt(xlvt_star)["theta"]
theta_dict_star = pecmy.rewrap_theta(theta_x_star)
for i in theta_dict.keys():
    np.savetxt(estimatesPath + i + ".csv", np.array([theta_dict_star[i]]), delimiter=",")
np.savetxt(estimatesPath + "v.csv", v_star, delimiter=",")

rcv_eq = pecmy.rcv_ft(v_star)
np.fill_diagonal(rcv_eq, 0)
np.savetxt(estimatesPath + "rcv_eq.csv", rcv_eq, delimiter=",")

cb_ratio = theta_dict_star["c_hat"] / rcv_eq
np.fill_diagonal(cb_ratio, 0)
cb_ratio_mean = np.sum(cb_ratio) / (pecmy.N - 1) ** 2
np.savetxt(estimatesPath + "cb_ratio_mean.csv", np.array([cb_ratio_mean]), delimiter=",")

### Compute Counterfactuals ###

if computeCounterfactuals == True:

    xlvt_star = np.genfromtxt(estimatesPath + 'x.csv', delimiter=',')
    xlvt_dict = pecmy.rewrap_xlvt(xlvt_star)
    theta_x_star = xlvt_dict["theta"]
    v_star = xlvt_dict["v"]

    xlvt_prime, obj, status = pecmy.estimator(v_star, theta_x_star, pecmy.mzeros, nash_eq=True)

    print(xlvt_prime)
    print(obj)
    print(status)

    xlvt_prime_path = counterfactualsPath + "x.csv"
    np.savetxt(xlvt_prime_path, xlvt_prime, delimiter=",")

print("done.")
