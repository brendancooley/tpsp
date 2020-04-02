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
import scipy.optimize as opt
import statsmodels.api as sm

import economy
import policies

import helpers_tpsp as hp
basePath = os.path.expanduser('~')

projectPath = basePath + "/Github/tpsp/"
projectFiles = basePath + "/Dropbox (Princeton)/1_Papers/tpsp/01_data/"

size = "mid/"

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

# v = np.ones(N)
# v = np.array([1.08, 1.65, 1.61, 1.05, 1.05, 1.30])
# v = np.repeat(1.4, N)

# TODO try just running inner loop, problem is that values of v change with theta as well, no reason we should run theta until covergence rather than iterating on v first.

imp.reload(policies)
imp.reload(economy)
pecmy = policies.policies(data, params, ROWname, results_path=resultsPath)  # generate pecmy and rcv vals
# np.seterr(all='raise')

# ccodes
# pecmy.ft_sv(6, np.ones(pecmy.x_len))

theta_dict = dict()
# theta_dict["c_hat"] = .25
theta_dict["eta"] = 1
theta_dict["c_hat"] = .5
theta_dict["alpha1"] = .2
theta_dict["gamma"] = .5
theta_dict["C"] = np.repeat(.5, pecmy.N)
theta_x = pecmy.unwrap_theta(theta_dict)

# pecmy.W ** - .75

v = (pecmy.v_max() - 1) / 2 + 1
# v = np.ones(pecmy.N)

pecmy.estimator_bounds(theta_x, v, "upper")

x, obj, status = pecmy.estimator(v, theta_x, pecmy.m, nash_eq=False)
x_dict = pecmy.rewrap_xlhvt(x)
ge_dict = pecmy.ecmy.rewrap_ge_dict(x_dict["ge_x"])

print(ge_dict["tau_hat"]*pecmy.ecmy.tau)
print("-----")

id = 0
v = (pecmy.v_max() - 1) / 2 + 1
ge_x_test = np.ones(pecmy.x_len)
ft_x = pecmy.ft_sv(id, np.ones(pecmy.x_len))
ft_h = ft_x[-pecmy.hhat_len:]

pecmy.peace_probs(ge_x_test, ft_h, id, pecmy.m, v, theta_dict)


def peace_probs_wrap_C(C, thres):

    Cinv = C ** -1
    chi_ji = -Cinv
    pr_peace = np.exp(chi_ji)

    out = pr_peace - thres

    return(out)

opt.root(peace_probs_wrap_C, .5, args=(.01, ))

def peace_probs_wrap_alpha(alpha, thres):

    chi_ji = -np.min(pecmy.W[pecmy.W>1])**(-1*alpha)
    pr_peace = np.exp(chi_ji)

    out = pr_peace - thres

    return(out)

opt.root(peace_probs_wrap_alpha, .5, args=(.99, ))


# pecmy.H(ge_x_test, ft_h, id, pecmy.m, v, theta_dict)
# pecmy.G_hat_tilde(ge_x_test, ft_h, id, pecmy.m, v, theta_dict)
#
# xlh_test = np.concatenate((ge_x_test, np.zeros(pecmy.lambda_i_len), ft_h))
# pecmy.Lzeros_i_cons(xlh_test, id, pecmy.m, v, theta_dict)
#
# pecmy.Lzeros_i_bounds(ge_x_test, id, "lower")
# pecmy.Lzeros_i_bounds(ge_x_test, id, "upper")
# np.reshape(pecmy.v_sv(id, ge_x_test, v)[0:pecmy.N**2], (pecmy.N, pecmy.N))*pecmy.ecmy.tau
#
# x, obj, status = pecmy.Lsolve_i_ipopt(id, pecmy.m, v, theta_dict)
# x_dict = pecmy.rewrap_xlh(x)
# ge_dict = pecmy.ecmy.rewrap_ge_dict(x_dict["ge_x"])
# print("tau:")
# print(ge_dict["tau_hat"]*pecmy.ecmy.tau)
# print("h:")
# print(x_dict["h"])
# print("peace probs:")
# peace_probs = pecmy.peace_probs(x_dict["ge_x"], x_dict["h"], id, pecmy.m, v, theta_dict)
# print(peace_probs)

# tau_hat_prime = [ge_dict["tau_hat"][j, ] if j != id else 1 / pecmy.ecmy.tau[j, ] for j in range(pecmy.N)]
# pecmy.ecmy.geq_solve(np.array(tau_hat_prime), np.ones(pecmy.N))
# rcx = pecmy.rcx(ge_dict["tau_hat"], x_dict["h"], id)
# pecmy.G_hat(rcx, v, id, all=True)
# pecmy.R_hat(pecmy.ecmy.rewrap_ge_dict(rcx), v)
# pecmy.G_hat(x_dict["ge_x"], v, id, all=True)
# pecmy.R_hat(ge_dict, v)



# np.reshape(np.repeat(np.max(pecmy.ecmy.tau, axis=1), pecmy.N), (pecmy.N, pecmy.N))
# pecmy.xlshvt_len
# chi_test = pecmy.chi(pecmy.m, theta_dict)
# chi_test[:,0]




# v = np.ones(pecmy.N)
# ft_x = pecmy.ft_sv(0, np.ones(pecmy.x_len))
# pecmy.r_v(np.repeat(.9, pecmy.N))
# pecmy.R_hat(pecmy.ecmy.rewrap_ge_dict(ft_x), np.repeat(.6, pecmy.N))
# pecmy.ecmy.rewrap_ge_dict(ft_x)
# pecmy.G_hat(ft_x, np.repeat(.6, pecmy.N), 0, 1)


# test_f = ag.grad(pecmy.wv_rcx)

# i = 0
# ge_x_sv = np.ones(pecmy.x_len)
# ft_id = pecmy.ecmy.rewrap_ge_dict(pecmy.ft_sv(i, ge_x_sv))
# h_sv_i = pecmy.ecmy.unwrap_ge_dict(pecmy.ecmy.geq_solve(ft_id["tau_hat"], np.ones(pecmy.N)))[-pecmy.hhat_len:]
#
# rcx = pecmy.rcx(np.ones((pecmy.N, pecmy.N)), h_sv_i, i)
# wv_i = pecmy.wv_rcx(rcx, i, pecmy.m, v, theta_dict)
# -1*pecmy.war_diffs(np.ones(pecmy.x_len), v, wv_i, i)

#
# def wv_rcx_wrap(theta_x):
#     theta_dict = pecmy.rewrap_theta(theta_x)
#     return(pecmy.wv_rcx(rcx, i, pecmy.mzeros, v, theta_dict))
#
# xlshvt_test = pecmy.estimator_sv(pecmy.mzeros, v, theta_x)
# test_f = pecmy.estimator_cons_jac_wrap(pecmy.mzeros)
# test = test_f(xlshvt_test, np.zeros(pecmy.xlshvt_len*pecmy.g_len))
# test_jac = np.reshape(test, (pecmy.g_len, pecmy.xlshvt_len))
# test_jac.shape
# np.sum(test_jac[:,-4:])
# pecmy.loss_grad(xlshvt_test, np.zeros(len(xlshvt_test)))[-4:]
# # pecmy.geq_ub()[0:pecmy.N**2] * pecmy.ecmy.tau.ravel()
#
# war_diffs_i_jac_f = ag.jacobian(pecmy.war_diffs_xlshvt)
# test = war_diffs_i_jac_f(xlshvt_test, i, pecmy.mzeros)
# test[:,-4:]
# Lzeros_i_jac_f = ag.jacobian(pecmy.Lzeros_i_xlshvt)
# test2 = Lzeros_i_jac_f(xlshvt_test, i, pecmy.mzeros)
# test2[:,-4:]
# comp_slack_i_jac_f = ag.jacobian(pecmy.comp_slack_xlshvt)
# test3 = comp_slack_i_jac_f(xlshvt_test, i, pecmy.mzeros)
# test3[:,-4:]
# h_diffs_i_jac_f = ag.jacobian(pecmy.h_diffs_xlshvt)
# test4 = h_diffs_i_jac_f(xlshvt_test, i)
# test4[:,-4:]
# v = pecmy.v_max() - pecmy.v_buffer
# v = np.array([1.10122687, 1.37060769, 1.99432529, 1.12005803, 0.89220011, 1.18619372])
#
# pecmy.rho(theta_dict)
# pecmy.chi(pecmy.mzeros, theta_dict)
#
# id = 1
# x, obj, status = pecmy.Lsolve_i_ipopt(id, pecmy.m, v, theta_dict)
# x_dict = pecmy.rewrap_lbda_i_x(x)
# print(pecmy.ecmy.rewrap_ge_dict(x_dict["ge_x"])["tau_hat"]*pecmy.ecmy.tau)

# fname = "out/est_mid_loqo_vmids.csv"
# x, obj, status = pecmy.estimator(v, theta_x, pecmy.m, nash_eq=False)
# # x, obj, status = pecmy.estimator(v, theta_x, np.zeros((pecmy.N, pecmy.N)), nash_eq=False)
# np.savetxt(fname, x, delimiter=",")
# x = np.genfromtxt(fname, delimiter=",")
#
# x_dict = pecmy.rewrap_xlshvt(x)
# ge_dict = pecmy.ecmy.rewrap_ge_dict(x_dict["ge_x"])
# s = np.reshape(x_dict["s"], (pecmy.N, pecmy.N))
# lbda = np.reshape(x_dict["lbda"], (pecmy.N, pecmy.lambda_i_len))
#
# print(ge_dict["tau_hat"]*pecmy.ecmy.tau)
# print("-----")
# print("v:")
# print(x_dict["v"])
# print("-----")
# theta_dict = pecmy.rewrap_theta(x_dict["theta"])
# print("theta:")
# print(theta_dict)
#
# print("G_hat:")
# print(pecmy.G_hat(x_dict["ge_x"], v, 0, all=True))
#
# print("R_hat:")
# print(pecmy.R_hat(ge_dict, v))
#
# for i in range(pecmy.N):
#
#     print("s_i")
#     print(s[i, ])
#
#     lbda_chi_i = pecmy.rewrap_lbda_i(lbda[i, ])["chi_i"]
#     print("lbda_chi_i:")
#     print(lbda_chi_i)
#
#     h_i = np.reshape(x_dict["h"], (pecmy.N, pecmy.hhat_len))[i, ]
#     print("h_i:")
#     print(h_i)
#
#     ft_i = pecmy.ft_sv(i, x_dict["ge_x"])
#     print("ft_i")
#     print(ft_i)
#     print("-----")

# ge_dict["tau_hat"] * pecmy.ecmy.tau
# id = 0
# h_i = np.reshape(x_dict["h"], (pecmy.N, pecmy.hhat_len))[id, ]
# s_i = s[id, ]
# lbda_chi_i = pecmy.rewrap_lbda_i(lbda[id, ])["chi_i"]
# s_i
# lbda_chi_i
# pecmy.G_hat(x_dict["ge_x"], x_dict["v"], 0, all=True)
# rcx_i = pecmy.rcx(ge_dict["tau_hat"], h_i, id)
# pecmy.G_hat(rcx_i, x_dict["v"], 0, all=True)
# pecmy.ecmy.rewrap_ge_dict(rcx_i)
# pecmy.wv_rcx(rcx_i, id, pecmy.m, x_dict["v"], theta_dict)


# testing:
# 1) multiple constraints for best response
    # NOTE this works fine for best responses and equilibrium, it's specific values that are giving us problems
    # gamma = 1, c_hat = .1, alpha = 0 (?), v=1
    # this converges in 29 iterations (starting at v_sv)
# 2) struggling with v = 1.25, gamma = .5, c_hat = .25, alpha = -.0001

# 1) starting values of v at 1.25
# 2) estimate at normal starting values
# 3) equilibrium computation at struggling values for 2
    # indeed, ipopt throws "restoration failed at these"
    # gamma = .03, c_hat = .47, alpha = .00016
# 4) same as 1) but lower bound on v to .75
    # gamma and a lot of the vs zoom to lower bound for some reason

# still have problems with corner taus
# vs are much better behaved when we start estimator ges at 1 (at least at the beginning)
    # contra equilibrium computations

# 4 just alpha


# problems:
    # R_hat, chi (gamma)
    # looks like r_hats get stuck negative sometimes (NOTE: this seems to be biggest probelm at the moment)
        # part of the issue with this is that then govs want to make welfare as small as possible.
        # and we can't get focs to be satisfied if Rhat is restricted to be positive
        # lower bounds on tau also a problem here
    # equilibrium converges fast when we start at v_sv and not well otherwise

# seems like it's only when we move the vs away from zero that we have issues
    # seems like it's actually heterogeneity in the vs, repeat 1.1. works fine (or perhaps just magnitudes)
    # maybe just an issue when proposed eq tau drops below v
    # this might also explain why we struggle when starting values are at ones (sure enough, this doesn't go through keeping everything else the same). But this is strange because estimator is actually struggling to get Russia values down, everything else goes to the right place
    # we do fine on best responses in this case though
        # but only when we start at v_sv, otherwise we struggle
        # but this still seems to be a corner solution problem

### estimator
    # just vs
        # converges in 170 iterations with v_sv, no R_hat clip, upper bound on v
        # 3) v_sv (bounds on, no clip)
        # 4) ones (bounds on, no clip) (performs much worse than 3), trying with v_buffer
            # this is because constraints end up on even at mzeros when v approaches v_max
            # with v_buffer this ends up going to same place as v_sv and starting slow descent from there


# 1) estimation with mumps
# 2) equilibrium at v mids (ones)
    # trying with larger v_step and mumps
    # no difference in performance versus pardiso
# 3) equilibrium at v_mids (v_sv)
    # both of these go to corners
    # trying with larger v_step
    # this helps a lot
# 4) larger v_step and pardiso (estimation)
    # larger v_step makes a huge difference
    # also starting v at v_max mid

### full estimator

# 1) derivative checker
    # no errors
# 4) turn off adaptive mu strategy
    # also start lambdas at zero
    # this ends up "solving to acceptable level"


# 1) just estimate alpha
    # OPTIMAL SOLUTION FOUND at alpha = .0002, loss = 17.9

# 2) 1-norm, decrease v lower bound
# 4) try with 1-norm for loss (just vs)

# 1-norm might be a mess because we keep hopping around minimum

# 1) smooth l-1 loss, sv=1, lb=.75
    # went to corner
# 2) estimate alpha, 2-norm, sv=1, lb=.75
# 3) back to 2-norm, sv=1, lb=.75
    # converges nicely in 140 iterations, loss = 6.2
# 4) everything, 2-norm, sv=1, lb=.75
    # gamma went to corner
