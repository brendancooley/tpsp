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

helpersPath = os.path.expanduser(projectPath + "source/")
sys.path.insert(1, helpersPath)

import helpers

class results:

    def __init__(self, location, size, bootstrap=False, bootstrap_id=1):

        location = location  # local, hpc
        size = size # mini/, mid/, large/

        basePath = os.path.expanduser('~')

        if location == "local":
            projectPath = basePath + "/Github/tpsp/"
        if location == "hpc":
            projectPath = basePath + "/tpsp/"

        if location == "local":
            projectFiles = basePath + "/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
        if location == "hpc":
            projectFiles = projectPath

        data_dir_base = projectFiles + "data/"
        results_dir_base = projectFiles + "results/"

        if bootstrap == True:
            self.dataPath = data_dir_base + size + str(bootstrap_id) + "/"
            resultsPath = results_dir_base + size
            self.estimatesPath = resultsPath + "estimates/" + str(bootstrap_id) + "/"
        else:
            self.dataPath = data_dir_base + size
            resultsPath = results_dir_base + size
            self.estimatesPath = resultsPath + "estimates/"

        self.counterfactualsPath = resultsPath + "counterfactuals/"
        self.xlhvt_star_path = self.estimatesPath + "x.csv"

        helpers.mkdir(resultsPath)
        helpers.mkdir(self.estimatesPath)
        helpers.mkdir(counterfactualsPath)

        # Economic Parameters
        beta = np.genfromtxt(dataPath + 'beta.csv', delimiter=',')
        theta = np.genfromtxt(dataPath + 'theta.csv', delimiter=',')
        mu = np.genfromtxt(dataPath + 'mu.csv', delimiter=',')
        nu = np.genfromtxt(dataPath + 'nu.csv', delimiter=',')

        self.params = {"beta":beta,"theta":theta,"mu":mu,"nu":nu}

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
        self.ROWname = str(ROWname)

        M = M / np.min(M)  # normalize milex
        W = dists

        N = len(Y)

        E = Eq + Ex

        self.data = {"tau":tau,"Xcif":Xcif,"Y":Y,"E":E,"r":r,"D":D,"W":W,"M":M, "ccodes":ccodes}  # Note: log distance

    def compute_estimates(self):

        pecmy = policies.policies(self.data, self.params, self.ROWname)

        # starting values
        theta_dict = dict()
        theta_dict["eta"] = 1.
        theta_dict["c_hat"] = 10.
        theta_dict["alpha1"] = 0.
        theta_dict["alpha2"] = 0.
        theta_dict["gamma"] = 0.
        theta_dict["C"] = np.repeat(10., pecmy.N)

        v = np.mean(pecmy.ecmy.tau, axis=1)
        theta_x_sv = pecmy.unwrap_theta(theta_dict)

        start_time = time.time()
        xlhvt_star, obj, status = pecmy.estimator(v, theta_x_sv, pecmy.m, nash_eq=False)
        print("--- Estimator converged in %s seconds ---" % (time.time() - start_time))

        print(xlhvt_star)
        print(obj)
        print(status)

        np.savetxt(self.xlhvt_star_path, xlhvt_star, delimiter=",")

    def unravel_estimates(self):

        pecmy = policies.policies(self.data, self.params, self.ROWname)

        xlhvt_star = np.genfromtxt(self.xlhvt_star_path, delimiter=",")
        ge_x_star = pecmy.rewrap_xlhvt(xlhvt_star)["ge_x"]
        tau_star = pecmy.ecmy.rewrap_ge_dict(ge_x_star)["tau_hat"] * pecmy.ecmy.tau
        X_star = pecmy.ecmy.rewrap_ge_dict(ge_x_star)["X_hat"] * pecmy.ecmy.Xcif
        np.savetxt(self.estimatesPath + "X_star.csv", X_star, delimiter=",")

        v_star = pecmy.rewrap_xlhvt(xlhvt_star)["v"]
        theta_x_star = pecmy.rewrap_xlhvt(xlhvt_star)["theta"]
        theta_dict_star = pecmy.rewrap_theta(theta_x_star)
        for i in theta_dict_star.keys():
            np.savetxt(self.estimatesPath + i + ".csv", np.array([theta_dict_star[i]]), delimiter=",")
        np.savetxt(self.estimatesPath + "v.csv", v_star, delimiter=",")

        G_star = pecmy.G_hat(ge_x_star, v_star, 0, all=True)
        rcv_eq = pecmy.rcv_ft(ge_x_star, v_star)
        np.fill_diagonal(rcv_eq, 0)
        np.savetxt(self.estimatesPath + "rcv_eq.csv", rcv_eq, delimiter=",")

        # probabilities of peace
        h = np.reshape(pecmy.rewrap_xlhvt(xlhvt_star)["h"], (pecmy.N, pecmy.hhat_len))
        peace_prob_mat = np.zeros((pecmy.N, pecmy.N))
        for i in range(pecmy.N):
            peace_probs_i = pecmy.peace_probs(ge_x_star, h[i, ], i, pecmy.m, v_star, theta_dict_star)[1]
            tick = 0
            for j in range(pecmy.N):
                if j not in [i, pecmy.ROW_id]:
                    peace_prob_mat[i, j] = peace_probs_i[tick]
                    tick += 1
                else:
                    peace_prob_mat[i, j] = 1

        np.savetxt(self.estimatesPath + "peace_probs.csv", peace_prob_mat, delimiter=",")

    def compute_counterfactuals(self):

        pecmy = policies.policies(self.data, self.params, self.ROWname)

        xlhvt_star = np.genfromtxt(self.xlhvt_star_path, delimiter=",")

        xlhvt_dict = pecmy.rewrap_xlhvt(xlhvt_star)
        theta_x_star = xlhvt_dict["theta"]
        v_star = xlhvt_dict["v"]

        xlhvt_prime, obj, status = pecmy.estimator(v_star, theta_x_star, pecmy.mzeros, nash_eq=True)

        print(xlhvt_prime)
        print(obj)
        print(status)

        xlhvt_prime_path = self.counterfactualsPath + "x.csv"
        np.savetxt(xlhvt_prime_path, xlhvt_prime, delimiter=",")

# bootstrap: https://github.com/brendancooley/tpsp/blob/f45862d123edce4ecaa026d7fa3947e3dc11cfb6/01_code/convexity_test.py
