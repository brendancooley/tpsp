import numpy as np
import os
import imp
import timeit
import time
import csv
import sys
import matplotlib.pyplot as plt

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

params = {"beta":beta,"theta":theta,"mu":mu,"nu":nu}

""# Data
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
W = np.log(dists+1)

N = len(Y)

E = Eq + Ex

data = {"tau":tau,"Xcif":Xcif,"Y":Y,"E":E,"r":r,"D":D,"W":W,"M":M, "ccodes":ccodes}  # Note: log distance

imp.reload(policies)
pecmy = policies.policies(data, params, ROWname, results_path=resultsPath, rcv_ft=rcv_ft)  # generate pecmy and rcv vals
id = 0

tau_hat = np.ones((pecmy.N, pecmy.N))
tau_hat[0, 1] = 2
ge_dict = pecmy.ecmy.geq_solve(tau_hat, np.ones(pecmy.N))

v_test = np.ones(pecmy.N)
v_test = v_test * 2

pecmy.ecmy.U_hat(ge_dict)
pecmy.r_v(ge_dict, v_test)
pecmy.R_hat(ge_dict, v_test)
pecmy.ecmy.tau

theta_dict_init = dict()
theta_dict_init["sigma_epsilon"] = 1
theta_dict_init["c_hat"] = .2
theta_dict_init["alpha"] = .5
theta_dict_init["gamma"] = 1

m = np.diag(M)

epsilon = np.zeros((pecmy.N, pecmy.N))
wv_m = pecmy.war_vals(b_test, m, theta_dict_init, epsilon) # calculate war values
ids_j = np.delete(np.arange(pecmy.N), id)
wv_m_i = wv_m[:,id][ids_j]

tau_v = np.tile(np.array([v_test]).transpose(), (1, pecmy.N))
np.fill_diagonal(tau_v, 1)

tau_hat_sv = np.ones((pecmy.N, pecmy.N))
tau_hat_v_sv = tau_v / pecmy.ecmy.tau
tau_hat_sv[id, ] = tau_hat_v_sv[id, ] + .01

ge_dict = pecmy.ecmy.geq_solve(tau_hat_sv, np.ones(pecmy.N))
ge_x_sv = pecmy.ecmy.unwrap_ge_dict(ge_dict)

test_x = pecmy.br(ge_x_sv, v_test, wv_m_i, id)
test_dict = pecmy.ecmy.rewrap_ge_dict(test_x)
test_dict
test_dict["tau_hat"] * pecmy.ecmy.tau

r_hat_id = []
t_vals = np.arange(0, 3, .1)
tau_v = np.tile(np.array([v_test]).transpose(), (1, pecmy.N))
np.fill_diagonal(tau_v, 1)
tau_hat_sv = np.ones((pecmy.N, pecmy.N))
for i in t_vals:
    tau_v[id, 0] = v_test[id] + i
    tau_hat_v_sv = tau_v / pecmy.ecmy.tau
    tau_hat_sv[id, 0] = tau_hat_v_sv[id, 0]
    ge_dict = pecmy.ecmy.geq_solve(tau_hat_sv, np.ones(pecmy.N))
    r_hat_id.append(pecmy.R_hat(ge_dict, v_test)[id])

plt.plot(t_vals, r_hat_id)

id_i = 1
nft_sv = pecmy.nft_sv(id_i, np.ones(pecmy.x_len))


rc_pols_x = pecmy.br_war_ji(nft_sv, v_test, id, id_i, full_opt=True)
rc_pols_dict = pecmy.ecmy.rewrap_ge_dict(rc_pols_x)
rc_pols_dict["tau_hat"] * pecmy.ecmy.tau
pecmy.G_hat(rc_pols_x, v_test)
pecmy.R_hat(rc_pols_dict, v_test)
pecmy.ecmy.U_hat(rc_pols_dict)

# TODO: regime change values still very large. Need to think about structure of objective.
    # this might be ok though because it's also easier to satisfy constraints






tau_hat_sv = ge_dict["tau_hat"]
tau_hat_sv[id] = tau_hat_nft[id] # start slightly above free trade
ge_dict_sv = pecmy.ecmy.geq_solve(tau_hat_sv, np.ones(pecmy.N))
ge_x_sv = pecmy.ecmy.unwrap_ge_dict(ge_dict_sv)

# test_x = pecmy.br(ge_x_sv, b_test, wv_m_i, id)
# test_dict = pecmy.ecmy.rewrap_ge_dict(test_x)
# test_dict
# test_dict["tau_hat"] * pecmy.ecmy.tau












test_x = pecmy.Lsolve(np.ones((pecmy.N, pecmy.N)), b_test, m, theta_dict_init, id)
test_dict = pecmy.ecmy.rewrap_ge_dict(test_x)
test_dict
test_dict["tau_hat"] * pecmy.ecmy.tau

v = np.array([1, 2])
v = np.array([v])
v
np.sum(np.array([[1, 2],[3,4]]), axis=1)

X = np.array([[1,2],[-1,2]])
X[X<0] = 0
