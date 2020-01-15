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
rcv_ft = False

runEstimates = False

# dataFiles = os.listdir("tpsp_data/")

basePath = os.path.expanduser('~')
projectPath = basePath + "/Dropbox (Princeton)/1_Papers/tpsp/01_data/"

if mini is True:
    dataPath = projectPath + "tpsp_data_mini/"
    resultsPath = projectPath + "results_mini/"
elif large is True:
    dataPath = projectPath + "tpsp_data_large/"
    resultsPath = projectPath + "results_large/"
# elif rcv_ft is True:
#     dataPath = projectPath + "tpsp_data_mini/"
#     resultsPath = projectPath + "results_rcv_ft/"
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
ROWname = np.genfromtxt(dataPath + 'ROWname.csv', delimiter=',', dtype="str")[0]

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
pecmy = policies.policies(data, params, b, ROWname, results_path=resultsPath, rcv_ft=rcv_ft)  # generate pecmy and rcv vals

theta_dict_init = dict()
theta_dict_init["sigma_epsilon"] = 1
theta_dict_init["c_hat"] = .2
theta_dict_init["alpha"] = .25
theta_dict_init["gamma"] = .3

b_init = np.repeat(.5, pecmy.N)
# b_init = np.array([.3, 2, 2, 1.2, 0, .3])

# b_init, theta_dict_sv = pecmy.import_results(resultsPath + "estimates_sv.csv")
# theta_dict_sv["c_hat"] = .2

estimatesPath = resultsPath + "estimates/"
helpers.mkdir(estimatesPath)
# out_test = pecmy.est_loop(b_init, theta_dict_init, est_c=False, c_step=.1, estimates_path=estimatesPath)
if runEstimates is True:
    out_test = pecmy.est_loop(b_init, theta_dict_init, est_c=True, c_step=.1, c_min=.1, estimates_path=estimatesPath)

ests_tilde = estimatesPath + "ests_0.csv"
b_tilde, theta_dict_tilde = pecmy.import_results(ests_tilde)

np.savetxt(resultsPath + "b_tilde.csv", b_tilde, delimiter=",")
for key in theta_dict_tilde.keys():
    np.savetxt(resultsPath + key + "_tilde.csv", np.array([theta_dict_tilde[key]]), delimiter=",")

# imp.reload(policies)
# pecmy = policies.policies(data, params, b, results_path=resultsPath)
# test = pecmy.affinity_fp(b_tilde, theta_dict_tilde, m)

counterfactualPath = resultsPath + "counterfactuals/"
helpers.mkdir(counterfactualPath)

### No militaries ###

if not os.path.exists(counterfactualPath + "tau_prime.csv"):
    m_prime = np.diag(M)
    affinity = np.zeros((pecmy.N, pecmy.N))
    ge_x_sv = np.ones(pecmy.ecmy.ge_x_len)
    equilibrium_prime = pecmy.nash_eq(ge_x_sv, b_tilde, theta_dict_tilde, m_prime, affinity)
    equilibrium_dict_prime = pecmy.ecmy.rewrap_ge_dict(equilibrium_prime)

    tau_prime = equilibrium_dict_prime["tau_hat"] * pecmy.ecmy.tau
    np.savetxt(counterfactualPath + "tau_prime.csv", tau_prime, delimiter=",")
    G_prime = pecmy.G_hat(equilibrium_prime, b_tilde)
    np.savetxt(counterfactualPath + "G_prime.csv", G_prime, delimiter=",")
    V_prime = pecmy.ecmy.U_hat(equilibrium_dict_prime)
    np.savetxt(counterfactualPath + "V_prime.csv", G_prime, delimiter=",")
else:
    tau_prime = np.genfromtxt(counterfactualPath + "tau_prime.csv", delimiter=",")
    print(tau_prime)
    equilibrium_dict_prime = pecmy.ecmy.geq_solve(tau_prime, np.ones(pecmy.N))
    V_prime = pecmy.ecmy.U_hat(equilibrium_dict_prime)
    np.savetxt(counterfactualPath + "V_prime.csv", V_prime, delimiter=",")

### best fit with militaries ###

# tau_prime = np.genfromtxt(counterfactualPath + "tau_prime.csv", delimiter=",")
if not os.path.exists(counterfactualPath + "tau_star.csv"):
    tau_hat_prime = tau_prime / pecmy.ecmy.tau
    ge_dict_prime = pecmy.ecmy.geq_solve(tau_hat_prime, np.ones(pecmy.N))
    ge_x_prime = pecmy.ecmy.unwrap_ge_dict(ge_dict_prime)

    m = pecmy.M / np.ones((pecmy.N, pecmy.N))
    m = m.T
    m[pecmy.ROW_id,:] = 0
    m[:,pecmy.ROW_id] = 0
    m[pecmy.ROW_id,pecmy.ROW_id] = 1
    print(m)
    print(theta_dict_tilde)

    affinity = np.zeros((pecmy.N, pecmy.N))
    equilibrium = pecmy.nash_eq(ge_x_prime, b_tilde, theta_dict_tilde, m, affinity)
    equilibrium_dict = pecmy.ecmy.rewrap_ge_dict(equilibrium)

    tau_star = equilibrium_dict["tau_hat"] * pecmy.ecmy.tau
    np.savetxt(counterfactualPath + "tau_star.csv", tau_star, delimiter=",")
    G_star = pecmy.G_hat(equilibrium, b_tilde)
    np.savetxt(counterfactualPath + "G_star.csv", G_star, delimiter=",")
else:
    tau_star = np.genfromtxt(counterfactualPath + "tau_star.csv", delimiter=",")
    print(tau_star)
    equilibrium_dict = pecmy.ecmy.geq_solve(tau_star, np.ones(pecmy.N))
    V_star = pecmy.ecmy.U_hat(equilibrium_dict)
    np.savetxt(counterfactualPath + "V_star.csv", V_star, delimiter=",")
