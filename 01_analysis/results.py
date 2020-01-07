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

rcvPath = resultsPath + "rcv.csv"

# Economic Parameters
beta = np.genfromtxt(dataPath + 'beta.csv', delimiter=',')
theta = np.genfromtxt(dataPath + 'theta.csv', delimiter=',')
mu = np.genfromtxt(dataPath + 'mu.csv', delimiter=',')
nu = np.genfromtxt(dataPath + 'nu.csv', delimiter=',')

# Military Parameters
alpha_0 = 0  # force gained (lost) in offensive operations, regardless of distance
alpha_1 = -.1   # extra force gained (lost) for every log km traveled
gamma = 1
c_hat = .2  # relative cost of war

params = {"beta":beta,"theta":theta,"mu":mu,"nu":nu, "alpha_0":alpha_0, "alpha_1":alpha_1, "c_hat":c_hat, "gamma":gamma}

# welfare weights
b = np.repeat(0, len(nu))

vars = {"b":b}

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

M = M / np.min(M)  # normalize milex
W = np.log(dists+1)

N = len(Y)

E = Eq + Ex

data = {"tau":tau,"Xcif":Xcif,"Y":Y,"E":E,"r":r,"D":D,"W":W,"M":M, "ccodes":ccodes}  # Note: log distance


# This will finish in 6.5 hours if each opt takes 75 seconds (13 governments)
# lots of nonsense negative numbers for Turkey's (11) best response for U.S. (12), replace with open/closed value instead

# br_11_12 = pecmy.br_war_ji(np.ones(pecmy.x_len), np.zeros(pecmy.N), 11, 12, full_opt=False)
# pecmy.rcv[0][11, 12] = pecmy.G_hat(br_11_12, np.zeros(pecmy.N), ids=np.array([11]))
# pecmy.rcv

# export regime change vals
# np.savetxt(resultsPath + "rcv0.csv", pecmy.rcv[0], delimiter=",")
# np.savetxt(resultsPath + "rcv1.csv", pecmy.rcv[1], delimiter=",")

# calculate free trade vals
# tau_hat_ft = 1 / pecmy.ecmy.tau
# ge_dict_ft = pecmy.ecmy.geq_solve(tau_hat_ft, np.ones(pecmy.N))
# ge_x_ft = pecmy.ecmy.unwrap_ge_dict(ge_dict_ft)
# G_hat_ft = pecmy.G_hat(ge_x_ft, np.zeros(pecmy.N))

# export free trade vals (b=0)
# np.savetxt("results/Ghatft.csv", G_hat_ft, delimiter=",")

imp.reload(policies)
# b_init, theta_dict_init = pecmy.import_results(resultsPath+"estimates_sv.csv")
pecmy = policies.policies(data, params, b, results_path=resultsPath)  # generate pecmy and rcv vals

theta_dict_init = dict()
theta_dict_init["sigma_epsilon"] = 1
theta_dict_init["c_hat"] = .2
theta_dict_init["alpha"] = .25
theta_dict_init["gamma"] = .3

b_init = np.repeat(.5, pecmy.N)
# b_init = np.array([.3, .8, 1, 1, 0, .3])

# b_init, theta_dict_sv = pecmy.import_results("estimates_sv.csv")
# theta_dict_sv["c_hat"] = .2

estimatesPath = resultsPath + "estimates/"
helpers.mkdir(estimatesPath)
# out_test = pecmy.est_loop(b_init, theta_dict_init, est_c=False, c_step=.1, estimates_path=estimatesPath)
# out_test = pecmy.est_loop(b_init, theta_dict_init, est_c=True, c_step=.1, c_min=.1, estimates_path=estimatesPath)
# pecmy.export_results(out_test, resultsPath + "c_hat20.csv")

#
# m = pecmy.M / np.ones((pecmy.N, pecmy.N))
# m = m.T
# m[pecmy.ROW_id,:] = 0
# m[:,pecmy.ROW_id] = 0
# m[pecmy.ROW_id,pecmy.ROW_id] = 1
#
# m_diag = np.diagonal(m)
# m_frac = m / m_diag
# pecmy.est_theta(b_init, m, theta_dict_init)





# start_time = time.time()
# m = M / np.ones_like(tau)
# m = m.T
# id = 0
# wv = pecmy.war_vals(b_init, m, theta_dict_init, np.zeros((pecmy.N, pecmy.N))) # calculate war values
# ids_j = np.delete(np.arange(pecmy.N), id)
# wv_i = wv[:,id][ids_j]
# ge_x_sv = pecmy.nft_sv(id)
# br = pecmy.br(ge_x_sv, b_init, wv_i, id, mil=False)
# print("--- %s seconds ---" % (time.time() - start_time))
