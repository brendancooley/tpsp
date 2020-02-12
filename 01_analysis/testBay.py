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
# elif rcv_ft is True:
#     dataPath = projectPath + "tpsp_data_mini/"
#     resultsPath = projectPath + "results_rcv_ft/"
else:
    dataPath = projectPath + "tpsp_data/"
    resultsPath = projectPath + "results/"
helpers.mkdir(resultsPath)

rcvPath = resultsPath + "rcv.csv"

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

theta_dict_init = dict()
theta_dict_init["c_hat"] = .32203
theta_dict_init["alpha"] = .001
theta_dict_init["gamma"] = 5

# TODO try just running inner loop, problem is that values of v change with theta as well, no reason we should run theta until covergence rather than iterating on v first.

imp.reload(policies)
pecmy = policies.policies(data, params, ROWname, results_path=resultsPath)  # generate pecmy and rcv vals

theta_x = pecmy.unwrap_theta(theta_dict_init)

xlvt_sv = np.concatenate((np.ones(pecmy.x_len), np.repeat(.01, pecmy.lambda_i_len*pecmy.N), np.ones(pecmy.N), theta_x))

gsb = pecmy.g_sparsity_bin(xlvt_sv)
pecmy.g_sparsity_idx(gsb)
pecmy.g_len


pecmy.xlvt_len
pecmy.g_len

pecmy.x_len+pecmy.lambda_i_len * pecmy.N
pecmy.g_sparsity()


pecmy.estimator_cons_jac(xlvt_sv, np.zeros(pecmy.g_len*pecmy.xlvt_len))


pecmy.xlvt_len * pecmy.g_len
pecmy.g_len

g_sparsity_indices_a1 = np.array(np.meshgrid(range(pecmy.g_len), range(pecmy.xlvt_len))).T.reshape(-1,2)
g_sparsity_indices1 = (g_sparsity_indices_a1[:,0], g_sparsity_indices_a1[:,1])



g_sparsity_indices_a2 = pecmy.g_sparsity()
g_sparsity_indices2 = (g_sparsity_indices_a2[:,1], g_sparsity_indices_a2[:,0])

xlvt_sv = np.ones(pecmy.xlvt_len)
pecmy.estimator_cons_jac(xlvt_sv, np.zeros(pecmy.xlvt_len*pecmy.g_len))



g_mat = np.reshape(np.repeat(False, pecmy.xlvt_len*pecmy.g_len), (pecmy.xlvt_len, pecmy.g_len))
g_mat[0, ].shape


ge_dict = pecmy.ecmy.rewrap_ge_dict(xlvt_dict["ge_x"])
ge_dict["tau_hat"] * pecmy.ecmy.tau
#
# pecmy.ecmy.tau
# pecmy.r_v(ge_dict, v_star)
# pecmy.R_hat(ge_dict, v_star)
# wv = pecmy.war_vals(v_star, pecmy.m, pecmy.rewrap_theta(theta_x_star))
# pecmy.Lzeros_i(np.concatenate((np.ones(pecmy.x_len), np.zeros(pecmy.lambda_i_len))), 0, v_star, wv[:,0])
#
# test = pecmy.ecmy.tau - np.diag(np.diag(pecmy.ecmy.tau))

pecmy.rhoM(theta_dict_init)
pecmy.chi(pecmy.m, theta_dict_init)

rcv_ft = pecmy.rcv_ft(v_star)
wv = rcv_ft - 275 / pecmy.chi(pecmy.m, theta_dict_init)
pecmy.war_vals(v_star, pecmy.m, theta_dict_init)
np.exp(-.004*pecmy.W)

x, obj, status = pecmy.estimator(v_star, theta_x_star, nash_eq=True)

print(x)
print(obj)
print(status)





ge_dict
1 / pecmy.ecmy.tau


len(pecmy.ecmy.geq_diffs(np.ones(pecmy.x_len)))
Lzeros_i_jac_f = ag.jacobian(pecmy.Lzeros_i_xlvt)
pecmy.Lzeros_i_xlvt(xlvt_star, 0)
Lzeros_i_jac = Lzeros_i_jac_f(xlvt_star, 0)
# Lzeros_i_jac[3, ]
Lzeros_i_jac
Lzeros_i_jac.shape
Lzeros_i_jac[61, ]
Lzeros_i_jac.ravel()

pecmy.chi(pecmy.m, pecmy.rewrap_theta(theta_x_star))
pecmy.war_vals(v_star, pecmy.m, pecmy.rewrap_theta(theta_x_star))

### test autograd and numpy clip

def f_test(x):
    return(np.clip(x, 0, np.inf))

def f_test_grad(x):
    f_test_grad_f = ag.grad(f_test)
    return(f_test_grad_f(x))

f_test_grad(-1.)

### dropbox testing



# test_x = pecmy.ft_sv(6, np.ones(pecmy.x_len))
# print(test_x)

# v = np.array([1.0303, 1.0977, 1.1353, 1.0214, 1.0000, 1.0143])
# id = 0
# # m = pecmy.M / np.ones((pecmy.N, pecmy.N))
# # m = m.T
# m = np.diag(pecmy.M)
# wv = pecmy.war_vals(v, m, theta_dict_init)
#
# pecmy.estimator_bounds("lower")
# theta_x_sv = pecmy.unwrap_theta(theta_dict_init)
# _x, obj, status = pecmy.estimator(v, theta_x_sv, nash_eq=True)
#
# print(_x)
# print(obj)
# print(status)

# _x, obj, status = pecmy.br_ipyopt(v, id, None)
#


# br = pecmy.br(pecmy.v_sv(id, np.ones(pecmy.x_len), v), v, wv[:,id], id)
# print(pecmy.ecmy.rewrap_ge_dict(br)["tau_hat"]*pecmy.ecmy.tau)

# _x, obj, status = pecmy.Lsolve_i_ipopt(id, v, wv[:,id])
#
# print(_x)
# print(obj)
# print(status)
#
# x_dict = pecmy.ecmy.rewrap_ge_dict(_x[0:pecmy.x_len])
# print(x_dict["tau_hat"]*pecmy.ecmy.tau)
# print("multipliers:")
# print(_x[pecmy.x_len:])

# _x, obj, status = pecmy.br_ipyopt(v, id, wv[:,id])
#
# print(_x)
# print(obj)
# print(status)
#
# x_dict = pecmy.ecmy.rewrap_ge_dict(_x)
# print(x_dict["tau_hat"]*pecmy.ecmy.tau)

# m = np.diag(pecmy.M)
#
# # v = np.ones(pecmy.N)
# test = pecmy.rewrap_lambda_i(np.ones(pecmy.lambda_i_len))
# len(test["tau_hat"])
# pecmy.lambda_i_len
# test2 = pecmy.unwrap_lambda_i(test)
# len(test2)
# pecmy.x_len
#
# wv = pecmy.war_vals(v, m, theta_dict_init, np.zeros((pecmy.N, pecmy.N)))
# out = pecmy.Lzeros_eq(v, wv)
# print(out)

# out = pecmy.Lzeros_min(v, theta_dict_init, mtd="SLSQP")
# print(out)

# x_lbda_theta_sv = np.zeros(pecmy.x_len+pecmy.lambda_i_len*pecmy.N+3+pecmy.N)
# x_lbda_theta_sv[0:pecmy.x_len] = 1
# x_lbda_theta_sv[pecmy.x_len+pecmy.lambda_i_len*pecmy.N:pecmy.x_len+pecmy.lambda_i_len*pecmy.N+pecmy.N] = np.ones(pecmy.N)
# x_lbda_theta_sv[pecmy.x_len+pecmy.lambda_i_len*pecmy.N+pecmy.N] = theta_dict_init["c_hat"]
# x_lbda_theta_sv[pecmy.x_len+pecmy.lambda_i_len*pecmy.N+pecmy.N+1] = theta_dict_init["alpha"]
# x_lbda_theta_sv[pecmy.x_len+pecmy.lambda_i_len*pecmy.N+pecmy.N+2] = theta_dict_init["gamma"]
#
# for i in range(pecmy.N):
#     print(np.sum(pecmy.Lzeros_i_wrap_jac(x_lbda_theta_sv, m, i, "lower")))
#     print(np.sum(pecmy.ecmy.geq_diffs_grad(x_lbda_theta_sv, "lower"), axis=1))




# wv = pecmy.war_vals(v, m, theta_dict_init, np.zeros((pecmy.N, pecmy.N)))
# # wv = np.zeros((pecmy.N, pecmy.N))
# id = 3
# print(wv[:,id])
#
# x = pecmy.Lsolve(v, m, theta_dict_init, id, enforce_geq=True)
# x_dict = pecmy.ecmy.rewrap_ge_dict(x)
# print(x_dict)
# print(x_dict["tau_hat"] * pecmy.ecmy.tau)
# pecmy.G_hat(x[0:pecmy.x_len], v, 0, all=True)
#
# x2 = pecmy.br(np.ones(pecmy.x_len), v, wv[:,id], id)
# x2_dict = pecmy.ecmy.rewrap_ge_dict(x2)
# print(x2_dict)
# print(pecmy.G_hat(x2[0:pecmy.x_len], v, 0, all=True))



# lambda_i = np.zeros(pecmy.lambda_i_len)
# len(lambda_i)
# lambda_i_dict = pecmy.rewrap_lambda_i(lambda_i)
# len(lambda_i_dict["h_hat"])
# pecmy.x_len
# pecmy.N**2 + 4*pecmy.N + pecmy.N**2
# pecmy.Lzeros_theta_min(theta_dict_init, np.ones(pecmy.N))



# pecmy.Lzeros_i_wrap(x_lbda_theta_sv, m, 0)
# start_time = time.time()
# pecmy.Lzeros_i_wrap_jac(x_lbda_theta_sv, m, 0)
# print("--- %s seconds ---" % (time.time() - start_time))
#
# start_time = time.time()
# pecmy.Lzeros_all_jac(x_lbda_theta_sv, m, "lower")
# print("--- %s seconds ---" % (time.time() - start_time))
#
# pecmy.Lzeros_theta_grad(theta_lbda_chi_init)
#
#
# id = 2
# L_grad_f = ag.grad(pecmy.Lagrange_i_x)
# v = np.ones(pecmy.N)
# v[id] = 1.3
# L_grad = L_grad_f(np.ones(pecmy.x_len), v, np.ones((pecmy.N, pecmy.N)), wv[:,id], np.zeros(pecmy.lambda_i_len), id)
#
# pecmy.ecmy.rewrap_ge_dict(L_grad)
#
# np.sum(L_grad)
#
# pecmy.rcv_ft(np.ones(pecmy.N))
#
# m = pecmy.M / np.ones((pecmy.N, pecmy.N))
# m = m.T
# m[pecmy.ROW_id,:] = 0
# m[:,pecmy.ROW_id] = 0
# m[pecmy.ROW_id,pecmy.ROW_id] = 1
#
# pecmy.chi_prime(m, theta_dict_init)
# pecmy.W
#
#
#
#
#
#
# pecmy.war_vals(v, m, theta_dict_init, np.zeros((pecmy.N, pecmy.N)))
#
#
#
# m_diag = np.diagonal(m)
# m_frac = m / m_diag
#
# # m = np.diag(M)
#
# id = 2
#
# testL = pecmy.Lsolve(v, m, theta_dict_init, id, mtd="lm")
# testL_dict = pecmy.ecmy.rewrap_ge_dict(testL)
# testL_dict
# pecmy.G_hat(testL, v, 0, all=True)
#
# ge_x_sv = pecmy.v_sv(id, np.ones(pecmy.x_len), v)
# war_vals = pecmy.war_vals(v, m, theta_dict_init, np.zeros((pecmy.N, pecmy.N)))
# testbr = pecmy.br(ge_x_sv, v, war_vals[:,id], id)
# testbr_dict = pecmy.ecmy.rewrap_ge_dict(testbr)
# testbr_dict
# pecmy.G_hat(testbr, v, 0, all=True)
#
# sv = np.concatenate((np.ones(pecmy.x_len), np.zeros(pecmy.lambda_i_len)))
# pecmy.Lzeros(sv, v, np.ones((pecmy.N, pecmy.N)), war_vals[:,id], id)
#
#
# test = pecmy.est_theta_inner(v, theta_dict_init, m)
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
#
# rcv = np.zeros((pecmy.N, pecmy.N))  # empty regime change value matrix (row's value for invading column)
# for i in range(pecmy.N):
#     v_nearest = hp.find_nearest(pecmy.v_vals, v[i])
#     rcv[i, ] = pecmy.rcv[v_nearest][i, ]  # grab rcvs associated with b_nearest and extract ith row
#
# m_diag = np.diagonal(m)
# m_frac = m / m_diag
#
# epsilon_star = pecmy.epsilon_star(v, m, theta_dict_init)
# t_epsilon = pecmy.trunc_epsilon(epsilon_star, theta_dict_init)
# phi = stats.norm.cdf(epsilon_star.ravel(), loc=0, scale=theta_dict_init["sigma_epsilon"])
# lhs = pecmy.Y(rcv, theta_dict_init, 1)
#
# X = np.column_stack((1-phi.ravel())*(np.log(m_frac.ravel()), (1-phi.ravel())*pecmy.W.ravel()))
# X.shape
# X_active = X[~np.isnan(X[:,0]),:]
# X_active.shape
# Y = lhs.ravel()
# Y.shape
# Y_active = Y[~np.isnan(X[:,0])]
# len(np.isnan(X[:,0]))
#
# m_diag = np.diagonal(m)
# m_frac = m / m_diag
#
# ests = sm.OLS(Y_active, X_active, missing="drop").fit()
# ests.params
#
#
# # pecmy.est_theta_inner(v_init, theta_dict_init, m)
#
# # pecmy.est_loop_interior(v_init, theta_dict_init)
# m = pecmy.M / np.ones((pecmy.N, pecmy.N))
# m = m.T
# wv = pecmy.war_vals(v, m, theta_dict_init, np.zeros((pecmy.N, pecmy.N)))
# rcv[:,2]
# wv
# G_lower = pecmy.G_lower(v, m, theta_dict_init)
#
#
#
#
#
# epsilon_star = pecmy.epsilon_star(v, m, theta_dict_init)
# Y_lower = pecmy.Y(rcv, theta_dict_init, G_lower)
# t_epsilon = pecmy.trunc_epsilon(epsilon_star, theta_dict_init)
#
# lhs = pecmy.Y(rcv, theta_dict_init, 1)
# phi = stats.norm.cdf(epsilon_star.ravel(), loc=0, scale=theta_dict_init["sigma_epsilon"])
#
# lhs.ravel() - Y_lower.ravel()
#
# Y = lhs.ravel() - (1 - phi.ravel()) * t_epsilon.ravel() - phi.ravel() * Y_lower.ravel()
# # NOTE: truncated epsilons are very very large....is this the right way to do this?
#
#
# X = np.column_stack((1-phi.ravel())*(np.log(m_frac.ravel()), (1-phi.ravel())*pecmy.W.ravel()))
# # X = np.column_stack((np.log(m_frac.ravel()), pecmy.W.ravel()))
# plt.plot(X[:,1], Y.ravel(), "+")
# plt.plot(X[:,0], Y.ravel(), "+")
# #
# # weights = 1 - phi
# #
# X_active = X[X[:,0]!=-np.inf,:]
# Y_active = lhs.ravel()[X[:,0]!=-np.inf]
# # weights_active = weights[X[:,0]!=-np.inf]
# Y_active = Y_active[~np.isnan(X_active[:,0])]
# # weights_active = weights_active[~np.isnan(X_active[:,0])]
# X_active = X_active[~np.isnan(X_active[:,0]),:]
# # len(Y)
# # len(X[:,0])
# #
# # pecmy.est_theta(X_active, Y_active)
#
# # test = sm.OLS(Y_active, X_active, missing="drop").fit()
# # test2 = sm.WLS(Y_active, X_active, missing="drop").fit()
# #
# # test.params
# # test2.params
# #
# #
# #
# #
# # ge_x_sv = pecmy.v_sv(2, np.ones(pecmy.x_len), v_init)
# # test_br1 = pecmy.br(ge_x_sv, v, wv[:,2], 2)
# # test_br1_dict = pecmy.ecmy.rewrap_ge_dict(test_br1)
# # test_br1_dict["tau_hat"] * pecmy.ecmy.tau
# # test_br2 = pecmy.br(ge_x_sv, v, np.zeros(pecmy.N), 2)
# # test_br2_dict = pecmy.ecmy.rewrap_ge_dict(test_br2)
# #
# # pecmy.G_hat(test_br1, v_init, 2)
# # pecmy.G_hat(test_br2, v_init, 2)
# # pecmy.R_hat(test_br1_dict, v_init)
# #
# # id = 2
# #
# # # m = np.diag(pecmy.M)
# #
# # v = np.array([1.1, 1.3, 1.9, 1.1, 1, 1.2])
# # epsilon = np.zeros((pecmy.N, pecmy.N))
# # wv_m = pecmy.war_vals(v, m, theta_dict_init, epsilon) # calculate war values
# # wv_m_i = wv_m[:,id]
# # wv_m_i
# #
# # v_sv = pecmy.v_sv(id, np.ones(pecmy.x_len), v)
# #
# # test_x = pecmy.br(v_sv, v, wv_m_i, id)
# # test_dict = pecmy.ecmy.rewrap_ge_dict(test_x)
# # test_dict["tau_hat"] * pecmy.ecmy.tau
# #
# # v_sv_0 = pecmy.v_sv(0, np.ones(pecmy.x_len), v)
# # test = pecmy.br_war_ji(v_sv_0, v, 4, 0, full_opt=True)
# # test_dict = pecmy.ecmy.rewrap_ge_dict(test)
# # test_dict
# #
# #
# # m = pecmy.M / np.ones((pecmy.N, pecmy.N))
# # m = m.T
# # # m = np.diag(pecmy.M)
# # v_test = np.ones(pecmy.N)
# # epsilon = np.zeros((pecmy.N, pecmy.N))
# # wv_m = pecmy.war_vals(v_test, m, theta_dict_init, epsilon) # calculate war values
# #
# #
# # # test = pecmy.est_v_i_grid(2, v_test, m, theta_dict_init, epsilon)
# # # test = pecmy.est_v_grid(v_test, m, theta_dict_init, epsilon)
# # epsilon_star_test = pecmy.epsilon_star(v_sv, m, theta_dict_init)
# # pecmy.trunc_epsilon(epsilon_star_test, theta_dict_init)
# #
# # epsilon_test = np.reshape(np.random.normal(0, theta_dict_init["sigma_epsilon"], pecmy.N ** 2), (pecmy.N, pecmy.N))
# # pecmy.est_theta_outer(v_sv, theta_dict_init)
# #
# #
# # theta_dict_init["gamma"] = test_params[0]
# # theta_dict_init["alpha"] = test_params[1]
# # rhoM = pecmy.rhoM(theta_dict_init, 0)
# # chi_test = np.zeros((pecmy.N, pecmy.N))
# # for i in range(pecmy.N):
# #     for j in range(pecmy.N):
# #         if i != j:
# #             v_j = v_sv[j]
# #             v_j_nearest = hp.find_nearest(pecmy.v_vals, v_j)
# #             rcv_ji = pecmy.rcv[v_j_nearest][j, i]  # get regime change value for j controlling i's policy
# #             m_x = pecmy.unwrap_m(m)
# #             chi_ji = pecmy.chi(m_x, j, i, theta_dict_init, rhoM)
# #             chi_test[j, i] = chi_ji
# #
# #
# #
# #
# #
# # # NOTE: increasing gamma increases epsilon star and moves trunc_epsilon, implying higher gamma...
# #
# # m = pecmy.M / np.ones((pecmy.N, pecmy.N))
# # m = m.T
# # m[pecmy.ROW_id,:] = 0
# # m[:,pecmy.ROW_id] = 0
# # m[pecmy.ROW_id,pecmy.ROW_id] = 1
# # m_diag = np.diagonal(m)
# # m_frac = m / m_diag
# #
# # rcv = np.zeros((pecmy.N, pecmy.N))  # empty regime change value matrix (row's value for invading column)
# # for i in range(pecmy.N):
# #     v_nearest = hp.find_nearest(pecmy.v_vals, v[i])
# #     rcv[i, ] = pecmy.rcv[v_nearest][i, ]  # grab rcvs associated with b_nearest and extract ith row
# #     # (i's value for invading all others)
# # # rcv = rcv.T
# # print("rcv: ")
# # print(rcv)
# #
# # epsilon_star = pecmy.epsilon_star(v_sv, m, theta_dict_init)
# # t_epsilon = pecmy.trunc_epsilon(epsilon_star, theta_dict_init)
# #
# # lhs = np.log( 1 / (theta_dict_init["c_hat"] ** -1 * (rcv - 1) - 1) )
# # Y = lhs.ravel() - t_epsilon.ravel()
# # X = np.column_stack((np.log(m_frac.ravel()), pecmy.W.ravel()))
# #
# # active_bin = epsilon_star < 0
# # epsilon_star
# # active_bin
# # indicator = active_bin.ravel()
# #
# # Y_active = Y[indicator]
# # X_active = X[indicator, ]
# #
# # plt.plot(X_active[:,1], Y_active, "r+")
# #
# #
# # theta_dict_init["alpha"] = .0001
# # theta_dict_init["gamma"] = 1
# # imp.reload(policies)
# # imp.reload(economy)
# # pecmy = policies.policies(data, params, ROWname, results_path=resultsPath, rcv_ft=rcv_ft)
# # pecmy.est_theta_inner(v_init, theta_dict_init, m)
# # #
# # #
# # #
# # #
# # # np.array([1, 2, 3])[np.array([True, False, False])]
# # #
# # #
# # #
# # #
# # # np.array([[0, 0],[1, 1]]) > np.array([[1, 1],[0, 0]])
# # # np.zeros((2, 3))
# # #
# # #
# # #
# # #
# # # tau_hat = np.ones((pecmy.N, pecmy.N))
# # # tau_hat[0, ] = 3
# # # tau_hat[0, 4] = 1 / pecmy.ecmy.tau[0, 4]
# # # tau_hat[0, 0] = 1
# # # ge_dict = pecmy.ecmy.geq_solve(tau_hat, np.ones(pecmy.N))
# # #
# # # v_test = np.ones(pecmy.N)
# # # v_test = v_test * 1.7
# # #
# # # pecmy.ecmy.U_hat(ge_dict)
# # # pecmy.r_v(ge_dict, v_test)
# # # pecmy.R_hat(ge_dict, v_test)
# # # pecmy.G_hat(pecmy.ecmy.unwrap_ge_dict(ge_dict), v_test)
# # # pecmy.ecmy.tau
# # #
# # #
# # #
# # #
# # #
# # # ids_j = np.delete(np.arange(pecmy.N), id)
# # # wv_m_i = wv_m[:,id][ids_j]
# # #
# # # tau_v = np.tile(np.array([v_test]).transpose(), (1, pecmy.N))
# # # np.fill_diagonal(tau_v, 1)
# # #
# # # tau_hat_sv = np.ones((pecmy.N, pecmy.N))
# # # tau_hat_v_sv = tau_v / pecmy.ecmy.tau
# # # tau_hat_sv[id, ] = tau_hat_v_sv[id, ] + .01
# # #
# # # ge_dict = pecmy.ecmy.geq_solve(tau_hat_sv, np.ones(pecmy.N))
# # # ge_x_sv = pecmy.ecmy.unwrap_ge_dict(ge_dict)
# # #
# # # test_x = pecmy.br(ge_x_sv, v_test, wv_m_i, id)
# # # test_dict = pecmy.ecmy.rewrap_ge_dict(test_x)
# # # test_dict
# # # test_dict["tau_hat"] * pecmy.ecmy.tau
# # #
# # # r_hat_id = []
# # # t_vals = np.arange(0, 3, .1)
# # # v_test[1] = 1.5
# # # tau_v = np.tile(np.array([v_test]).transpose(), (1, pecmy.N))
# # # np.fill_diagonal(tau_v, 1)
# # # tau_hat_sv = np.ones((pecmy.N, pecmy.N))
# # # for i in t_vals:
# # #     tau_v[id, 0] = v_test[id] + i
# # #     tau_hat_v_sv = tau_v / pecmy.ecmy.tau
# # #     tau_hat_sv[id, 0] = tau_hat_v_sv[id, 0]
# # #     ge_dict = pecmy.ecmy.geq_solve(tau_hat_sv, np.ones(pecmy.N))
# # #     r_hat_id.append(pecmy.R_hat(ge_dict, v_test)[id])
# # #
# # # plt.plot(t_vals, r_hat_id)
# # #
# # # id_i = 0
# # # nft_sv = pecmy.nft_sv(id_i, np.ones(pecmy.x_len))
# # #
# # #
# # # v_test[id] = 1.69
# # # rc_pols_x = pecmy.br_war_ji(nft_sv, v_test, id, id_i, full_opt=True)
# # # rc_pols_dict = pecmy.ecmy.rewrap_ge_dict(rc_pols_x)
# # # rc_pols_dict["tau_hat"] * pecmy.ecmy.tau
# # # pecmy.G_hat(rc_pols_x, v_test)
# # # pecmy.R_hat(rc_pols_dict, v_test)
# # # pecmy.ecmy.U_hat(rc_pols_dict)
# # # pecmy.ecmy.tau
# # #
# # # # TODO: regime change values still very large. Need to think about structure of objective.
# # #     # this might be ok though because it's also easier to satisfy constraints
# # #
# # #
# # #
# # #
# # #
# # #
# # # tau_hat_sv = ge_dict["tau_hat"]
# # # tau_hat_sv[id] = tau_hat_nft[id] # start slightly above free trade
# # # ge_dict_sv = pecmy.ecmy.geq_solve(tau_hat_sv, np.ones(pecmy.N))
# # # ge_x_sv = pecmy.ecmy.unwrap_ge_dict(ge_dict_sv)
# # #
# # # # test_x = pecmy.br(ge_x_sv, b_test, wv_m_i, id)
# # # # test_dict = pecmy.ecmy.rewrap_ge_dict(test_x)
# # # # test_dict
# # # # test_dict["tau_hat"] * pecmy.ecmy.tau
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # # test_x = pecmy.Lsolve(np.ones((pecmy.N, pecmy.N)), b_test, m, theta_dict_init, id)
# # # test_dict = pecmy.ecmy.rewrap_ge_dict(test_x)
# # # test_dict
# # # test_dict["tau_hat"] * pecmy.ecmy.tau
# # #
# # # v = np.array([1, 2])
# # # v = np.array([v])
# # # v
# # # np.sum(np.array([[1, 2],[3,4]]), axis=1)
# # #
# # # X = np.array([[1,2],[-1,2]])
# # # X[X<0] = 0
