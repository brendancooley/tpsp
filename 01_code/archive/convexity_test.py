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

# import multiprocessing as mp
# print("Number of processors: ", mp.cpu_count())

import threading
import logging

basePath = os.path.expanduser('~')
# location = sys.argv[1]  # local, hpc
# size = sys.argv[2] # mini/, mid/, large/
location = "local"
size = "mini/"

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

data_dir_base = projectFiles + "data/"
results_dir_base = projectFiles + "results/"

dataPath = data_dir_base + size
resultsPath_test = results_dir_base + "test/"
resultsPath = resultsPath_test + size

helpers.mkdir(resultsPath_test)
helpers.mkdir(resultsPath)

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

lossPath = resultsPath + "loss/"
xPath = resultsPath + "x/"

helpers.mkdir(lossPath)
helpers.mkdir(xPath)

pecmy = policies.policies(data, params, ROWname, resultsPath)
B = 5

def eq_loss(theta_dict, v, id):
    logging.info("Thread starting " + str(id))
    theta_x = pecmy.unwrap_theta(theta_dict)
    x, obj, status = pecmy.estimator(v, theta_x, pecmy.m, nash_eq=True)
    loss = pecmy.loss(x)
    np.savetxt(xPath + str(id) + ".csv", x, delimiter=",")
    np.savetxt(lossPath + str(id) + ".csv", np.array([loss]), delimiter=",")
    logging.info("Thread finishing " + str(id))

pecmy.loss(np.ones(pecmy.xlvt_len))

alpha1_vec = np.linspace(-pecmy.alpha1_ub, pecmy.alpha1_ub, B)
v = np.array([1.2, 1.4, 1.9, 1.1, 1., 1.25])
theta_dict = dict()
theta_dict["c_hat"] = .5
theta_dict["alpha0"] = 0
theta_dict["gamma"] = 1

if __name__ == '__main__':

    threads = list()
    # pool = mp.Pool(mp.cpu_count())
    for i in range(len(alpha1_vec)):
        theta_dict["alpha1"] = alpha1_vec[i]
        f = threading.Thread(target=eq_loss, args=(theta_dict, v, i, ))
        threads.append(f)
        f.start()
        # pool.apply_async(pecmy.estimator, args=(v, theta_dict, pecmy.m, True))

    for index, thread in enumerate(threads):
        logging.info("Main    : before joining thread %d.", index)
        thread.join()
        logging.info("Main    : thread %d done", index)

    # pool.close()
    # pool.join()

    print("done.")


### TESTING ###

# failed restoration phase
# infeasibility with 2 is in P_hat, same with 4
x_sv = np.genfromtxt(xPath + "4.csv", delimiter=",")

ge_x = pecmy.rewrap_xlvt(x_sv)["ge_x"]
ge_dict = pecmy.ecmy.rewrap_ge_dict(ge_x)
theta_x = pecmy.rewrap_xlvt(x_sv)["theta"]

pecmy.ecmy.P_hat(ge_dict)
pecmy.ecmy.geq_solve(ge_dict["tau_hat"], np.ones(pecmy.N))
pecmy.ecmy.tau * ge_dict["tau_hat"]



cons = pecmy.estimator_cons(x_sv, pecmy.m)
geq = cons[0:pecmy.hhat_len]
geqX = geq[0:pecmy.N**2]
P = geq[pecmy.N**2:pecmy.N**2+pecmy.N]

Lzeros_len = len(pecmy.Lzeros_i_xlvt(x_sv, 0, pecmy.m))
Lzeros = cons[pecmy.hhat_len:Lzeros_len*pecmy.N]
war_diffs = cons[pecmy.hhat_len+Lzeros_len*pecmy.N:pecmy.hhat_len+Lzeros_len*pecmy.N+pecmy.N**2]
comp_slack = cons[pecmy.hhat_len+Lzeros_len*pecmy.N+pecmy.N**2:]



pecmy.estimator_cons_jac(x_sv, np.repeat(True, pecmy.g_len*pecmy.xlvt_len), pecmy.m)

v = pecmy.rewrap_xlvt(x_sv)["v"]
theta_x = pecmy.rewrap_xlvt(x_sv)["theta"]
lbda = np.reshape(pecmy.rewrap_xlvt(x_sv)["lbda"], (pecmy.N, pecmy.lambda_i_len))
pecmy.rewrap_lbda_i(lbda[0, ])

x, obj, status = pecmy.estimator(v, theta_x, pecmy.mzeros, nash_eq=True)
loss = pecmy.loss(x)
