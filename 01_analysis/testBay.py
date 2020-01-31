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
# W = np.log(dists+1)
W = dists

N = len(Y)

E = Eq + Ex

data = {"tau":tau,"Xcif":Xcif,"Y":Y,"E":E,"r":r,"D":D,"W":W,"M":M, "ccodes":ccodes}  # Note: log distance

theta_dict_init = dict()
theta_dict_init["sigma_epsilon"] = 1
theta_dict_init["c_hat"] = .2
theta_dict_init["alpha"] = .0005
theta_dict_init["gamma"] = 1

# TODO try just running inner loop, problem is that values of v change with theta as well, no reason we should run theta until covergence rather than iterating on v first.

imp.reload(policies)
imp.reload(economy)
pecmy = policies.policies(data, params, ROWname, results_path=resultsPath, rcv_ft=rcv_ft)  # generate pecmy and rcv vals

v_init = np.array([1.1, 1.3, 1.9, 1.1, 1, 1.2])
# pecmy.est_loop_interior(v_init, theta_dict_init)






id = 2
m = pecmy.M / np.ones((pecmy.N, pecmy.N))
m = m.T
# m = np.diag(pecmy.M)

v = np.array([1.1, 1.3, 1.9, 1.1, 1, 1.2])
epsilon = np.zeros((pecmy.N, pecmy.N))
wv_m = pecmy.war_vals(v, m, theta_dict_init, epsilon) # calculate war values
wv_m_i = wv_m[:,id]
wv_m_i

v_sv = pecmy.v_sv(id, np.ones(pecmy.x_len), v)

pecmy.G_hat_grad(v_sv, v, id, -1)
test_x = pecmy.br(v_sv, v, wv_m_i, id)
test_dict = pecmy.ecmy.rewrap_ge_dict(test_x)
test_dict["tau_hat"] * pecmy.ecmy.tau

v_sv_0 = pecmy.v_sv(0, np.ones(pecmy.x_len), v)
test = pecmy.br_war_ji(v_sv_0, v, 4, 0, full_opt=True)
test_dict = pecmy.ecmy.rewrap_ge_dict(test)
test_dict


m = pecmy.M / np.ones((pecmy.N, pecmy.N))
m = m.T
# m = np.diag(pecmy.M)
v_test = np.ones(pecmy.N)
epsilon = np.zeros((pecmy.N, pecmy.N))
wv_m = pecmy.war_vals(v_test, m, theta_dict_init, epsilon) # calculate war values


# test = pecmy.est_v_i_grid(2, v_test, m, theta_dict_init, epsilon)
# test = pecmy.est_v_grid(v_test, m, theta_dict_init, epsilon)
epsilon_star_test = pecmy.epsilon_star(v_sv, m, theta_dict_init)
pecmy.trunc_epsilon(epsilon_star_test, theta_dict_init)

epsilon_test = np.reshape(np.random.normal(0, theta_dict_init["sigma_epsilon"], pecmy.N ** 2), (pecmy.N, pecmy.N))
pecmy.est_theta_outer(v_sv, theta_dict_init)


theta_dict_init["gamma"] = test_params[0]
theta_dict_init["alpha"] = test_params[1]
rhoM = pecmy.rhoM(theta_dict_init, 0)
chi_test = np.zeros((pecmy.N, pecmy.N))
for i in range(pecmy.N):
    for j in range(pecmy.N):
        if i != j:
            v_j = v_sv[j]
            v_j_nearest = hp.find_nearest(pecmy.v_vals, v_j)
            rcv_ji = pecmy.rcv[v_j_nearest][j, i]  # get regime change value for j controlling i's policy
            m_x = pecmy.unwrap_m(m)
            chi_ji = pecmy.chi(m_x, j, i, theta_dict_init, rhoM)
            chi_test[j, i] = chi_ji





# NOTE: increasing gamma increases epsilon star and moves trunc_epsilon, implying higher gamma...

m = pecmy.M / np.ones((pecmy.N, pecmy.N))
m = m.T
m[pecmy.ROW_id,:] = 0
m[:,pecmy.ROW_id] = 0
m[pecmy.ROW_id,pecmy.ROW_id] = 1
m_diag = np.diagonal(m)
m_frac = m / m_diag

rcv = np.zeros((pecmy.N, pecmy.N))  # empty regime change value matrix (row's value for invading column)
for i in range(pecmy.N):
    v_nearest = hp.find_nearest(pecmy.v_vals, v[i])
    rcv[i, ] = pecmy.rcv[v_nearest][i, ]  # grab rcvs associated with b_nearest and extract ith row
    # (i's value for invading all others)
# rcv = rcv.T
print("rcv: ")
print(rcv)

epsilon_star = pecmy.epsilon_star(v_sv, m, theta_dict_init)
t_epsilon = pecmy.trunc_epsilon(epsilon_star, theta_dict_init)

lhs = np.log( 1 / (theta_dict_init["c_hat"] ** -1 * (rcv - 1) - 1) )
Y = lhs.ravel() - t_epsilon.ravel()
X = np.column_stack((np.log(m_frac.ravel()), pecmy.W.ravel()))

active_bin = epsilon_star < 0
epsilon_star
active_bin
indicator = active_bin.ravel()

Y_active = Y[indicator]
X_active = X[indicator, ]

plt.plot(X_active[:,1], Y_active, "r+")


theta_dict_init["alpha"] = .0001
theta_dict_init["gamma"] = 1
imp.reload(policies)
imp.reload(economy)
pecmy = policies.policies(data, params, ROWname, results_path=resultsPath, rcv_ft=rcv_ft)
pecmy.est_theta_inner(v_init, theta_dict_init, m)
#
#
#
#
# np.array([1, 2, 3])[np.array([True, False, False])]
#
#
#
#
# np.array([[0, 0],[1, 1]]) > np.array([[1, 1],[0, 0]])
# np.zeros((2, 3))
#
#
#
#
# tau_hat = np.ones((pecmy.N, pecmy.N))
# tau_hat[0, ] = 3
# tau_hat[0, 4] = 1 / pecmy.ecmy.tau[0, 4]
# tau_hat[0, 0] = 1
# ge_dict = pecmy.ecmy.geq_solve(tau_hat, np.ones(pecmy.N))
#
# v_test = np.ones(pecmy.N)
# v_test = v_test * 1.7
#
# pecmy.ecmy.U_hat(ge_dict)
# pecmy.r_v(ge_dict, v_test)
# pecmy.R_hat(ge_dict, v_test)
# pecmy.G_hat(pecmy.ecmy.unwrap_ge_dict(ge_dict), v_test)
# pecmy.ecmy.tau
#
#
#
#
#
# ids_j = np.delete(np.arange(pecmy.N), id)
# wv_m_i = wv_m[:,id][ids_j]
#
# tau_v = np.tile(np.array([v_test]).transpose(), (1, pecmy.N))
# np.fill_diagonal(tau_v, 1)
#
# tau_hat_sv = np.ones((pecmy.N, pecmy.N))
# tau_hat_v_sv = tau_v / pecmy.ecmy.tau
# tau_hat_sv[id, ] = tau_hat_v_sv[id, ] + .01
#
# ge_dict = pecmy.ecmy.geq_solve(tau_hat_sv, np.ones(pecmy.N))
# ge_x_sv = pecmy.ecmy.unwrap_ge_dict(ge_dict)
#
# test_x = pecmy.br(ge_x_sv, v_test, wv_m_i, id)
# test_dict = pecmy.ecmy.rewrap_ge_dict(test_x)
# test_dict
# test_dict["tau_hat"] * pecmy.ecmy.tau
#
# r_hat_id = []
# t_vals = np.arange(0, 3, .1)
# v_test[1] = 1.5
# tau_v = np.tile(np.array([v_test]).transpose(), (1, pecmy.N))
# np.fill_diagonal(tau_v, 1)
# tau_hat_sv = np.ones((pecmy.N, pecmy.N))
# for i in t_vals:
#     tau_v[id, 0] = v_test[id] + i
#     tau_hat_v_sv = tau_v / pecmy.ecmy.tau
#     tau_hat_sv[id, 0] = tau_hat_v_sv[id, 0]
#     ge_dict = pecmy.ecmy.geq_solve(tau_hat_sv, np.ones(pecmy.N))
#     r_hat_id.append(pecmy.R_hat(ge_dict, v_test)[id])
#
# plt.plot(t_vals, r_hat_id)
#
# id_i = 0
# nft_sv = pecmy.nft_sv(id_i, np.ones(pecmy.x_len))
#
#
# v_test[id] = 1.69
# rc_pols_x = pecmy.br_war_ji(nft_sv, v_test, id, id_i, full_opt=True)
# rc_pols_dict = pecmy.ecmy.rewrap_ge_dict(rc_pols_x)
# rc_pols_dict["tau_hat"] * pecmy.ecmy.tau
# pecmy.G_hat(rc_pols_x, v_test)
# pecmy.R_hat(rc_pols_dict, v_test)
# pecmy.ecmy.U_hat(rc_pols_dict)
# pecmy.ecmy.tau
#
# # TODO: regime change values still very large. Need to think about structure of objective.
#     # this might be ok though because it's also easier to satisfy constraints
#
#
#
#
#
#
# tau_hat_sv = ge_dict["tau_hat"]
# tau_hat_sv[id] = tau_hat_nft[id] # start slightly above free trade
# ge_dict_sv = pecmy.ecmy.geq_solve(tau_hat_sv, np.ones(pecmy.N))
# ge_x_sv = pecmy.ecmy.unwrap_ge_dict(ge_dict_sv)
#
# # test_x = pecmy.br(ge_x_sv, b_test, wv_m_i, id)
# # test_dict = pecmy.ecmy.rewrap_ge_dict(test_x)
# # test_dict
# # test_dict["tau_hat"] * pecmy.ecmy.tau
#
#
#
#
#
#
#
#
#
#
#
#
# test_x = pecmy.Lsolve(np.ones((pecmy.N, pecmy.N)), b_test, m, theta_dict_init, id)
# test_dict = pecmy.ecmy.rewrap_ge_dict(test_x)
# test_dict
# test_dict["tau_hat"] * pecmy.ecmy.tau
#
# v = np.array([1, 2])
# v = np.array([v])
# v
# np.sum(np.array([[1, 2],[3,4]]), axis=1)
#
# X = np.array([[1,2],[-1,2]])
# X[X<0] = 0
