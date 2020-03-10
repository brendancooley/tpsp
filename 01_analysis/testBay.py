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
basePath = os.path.expanduser('~')

projectPath = basePath + "/Github/tpsp/"
projectFiles = basePath + "/Dropbox (Princeton)/1_Papers/tpsp/01_data/"

size = "mini/"

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

theta_dict = dict()
theta_dict["c_hat"] = .2
theta_dict["alpha0"] = 0
theta_dict["alpha1"] = 0
theta_dict["gamma"] = .5

# v = np.ones(N)
# v = np.array([1.08, 1.65, 1.61, 1.05, 1.05, 1.30])
# v = np.repeat(1.4, N)

# TODO try just running inner loop, problem is that values of v change with theta as well, no reason we should run theta until covergence rather than iterating on v first.

imp.reload(policies)
pecmy = policies.policies(data, params, ROWname, results_path=resultsPath)  # generate pecmy and rcv vals
theta_x = pecmy.unwrap_theta(theta_dict)

pecmy.estimator_bounds(theta_x, np.ones(pecmy.N), bound="upper")

# v = np.repeat(1.05, pecmy.N)
v = (pecmy.v_max() - 1) / 2 + 1
# v = np.array([1.03, 1.08, 1.04, 1.06, 1.02, 1.00])
# v = np.ones(pecmy.N)

# ge_v_sv = pecmy.v_sv_all(v)
# ft_sv = pecmy.ecmy.geq_solve(1 / pecmy.ecmy.tau, np.ones(pecmy.N))
# ge_dict = pecmy.ecmy.rewrap_ge_dict(ge_v_sv)
# pecmy.R_hat(ft_sv, v)
# pecmy.r_v(v)
# pecmy.ecmy.tau
# pecmy.x_len + pecmy.lambda_i_len*pecmy.N + pecmy.N**2 + pecmy.N**2
# pecmy.x_len + pecmy.lambda_i_len*pecmy.N + pecmy.N**2 + pecmy.hhat_len

# pecmy.ecmy.tau
# np.max(pecmy.ecmy.tau)
# pecmy.rho(theta_dict)
# pecmy.chi(pecmy.m, theta_dict)
# ft_id = pecmy.ft_sv(id, np.ones(pecmy.x_len))
# pecmy.G_hat(ft_id, v, id, all=True)
# pecmy.R_hat(pecmy.ecmy.rewrap_ge_dict(ft_id), v)
# pecmy.wv_rcx(ft_id, id, pecmy.m, v, theta_dict)

# id = 1
# x, obj, status = pecmy.Lsolve_i_ipopt(id, pecmy.m, v, theta_dict)
# x_dict = pecmy.rewrap_lbda_i_x(x)
# print(pecmy.ecmy.rewrap_ge_dict(x_dict["ge_x"])["tau_hat"]*pecmy.ecmy.tau)

x, obj, status = pecmy.estimator(v, theta_x, pecmy.mzeros, nash_eq=False)
x_dict = pecmy.rewrap_xlshvt(x)
ge_dict = pecmy.ecmy.rewrap_ge_dict(x_dict["ge_x"])


print(ge_dict["tau_hat"]*pecmy.ecmy.tau)
print("-----")
print("v:")
print(x_dict["v"])
print("-----")
theta_dict = pecmy.rewrap_theta(x_dict["theta"])
print("theta:")
print(theta_dict)

for i in range(pecmy.N):
    h_i = np.reshape(x_dict["h"], (pecmy.N, pecmy.hhat_len))[i, ]
    print(h_i)

    ft_i = pecmy.ft_sv(i, x_dict["ge_x"])
    print(ft_i)
    print("-----")




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
