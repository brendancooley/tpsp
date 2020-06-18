import numpy as np
import os
import imp
import timeit
import time
import csv
import sys

import c_economy as economy
import c_policies as policies
import c_setup as setup
import s_helpers_tpsp as hp

imp.reload(setup)

class results:

    def __init__(self, location, size, sv=None, bootstrap=False, bootstrap_id=0, mil_off=False):

        self.bootstrap_id = bootstrap_id
        self.setup = setup.setup(location, size, bootstrap=bootstrap, bootstrap_id=bootstrap_id, mil_off=mil_off)

        self.sv = sv
        self.mil_off = mil_off

        # Economic Parameters
        beta = np.genfromtxt(self.setup.beta_path, delimiter=',')
        theta = np.genfromtxt(self.setup.theta_path, delimiter=',')
        mu = np.genfromtxt(self.setup.mu_path, delimiter=',')
        nu = np.genfromtxt(self.setup.nu_path, delimiter=',')

        self.params = {"beta":beta,"theta":theta,"mu":mu,"nu":nu}

        # Data
        tau = np.genfromtxt(self.setup.tau_path, delimiter=',')
        Xcif = np.genfromtxt(self.setup.Xcif_path, delimiter=',')
        Y = np.genfromtxt(self.setup.Y_path, delimiter=',')
        Eq = np.genfromtxt(self.setup.Eq_path, delimiter=',')
        Ex = np.genfromtxt(self.setup.Ex_path, delimiter=',')
        r = np.genfromtxt(self.setup.r_path, delimiter=',')
        D = np.genfromtxt(self.setup.D_path, delimiter=',')
        ccodes = np.genfromtxt(self.setup.ccodes_path, delimiter=',', dtype="str")
        dists = np.genfromtxt(self.setup.dists_path, delimiter=',')
        M = np.genfromtxt(self.setup.M_path, delimiter=",")
        ROWname = np.genfromtxt(self.setup.ROWname_path, delimiter=',', dtype="str")

        self.ROWname = str(ROWname)

        M = M / np.min(M)  # normalize milex
        W = dists

        self.N = len(Y)

        E = Eq + Ex

        self.data = {"tau":tau,"Xcif":Xcif,"Y":Y,"E":E,"r":r,"D":D,"W":W,"M":M, "ccodes":ccodes}  # Note: log distance

    def compute_estimates(self):

        pecmy = policies.policies(self.data, self.params, self.ROWname, self.bootstrap_id)

        # starting values
        theta_dict = dict()
        theta_dict["eta"] = 1.
        theta_dict["c_hat"] = 25.
        theta_dict["alpha1"] = -.25
        theta_dict["alpha2"] = .5
        theta_dict["gamma"] = 1.
        theta_dict["C"] = np.repeat(25., pecmy.N)

        v = np.mean(pecmy.ecmy.tau, axis=1)
        theta_x_sv = pecmy.unwrap_theta(theta_dict)

        start_time = time.time()

        if self.mil_off == False:
            xlhvt_star, obj, status = pecmy.estimator(v, theta_x_sv, pecmy.m, sv=self.sv, nash_eq=False)
        else:
            xlhvt_star, obj, status = pecmy.estimator(v, theta_x_sv, np.diag(np.ones(pecmy.N)), sv=self.sv, nash_eq=False)

        if status == 0:

            print("--- Estimator converged in %s seconds ---" % (time.time() - start_time))
            print(xlhvt_star)
            print(obj)
            print(status)

            np.savetxt(self.setup.xlhvt_star_path, xlhvt_star, delimiter=",")
            sys.stdout.flush()

        else:
            print("estimator failed to converge after %s seconds" % (time.time() - start_time))
            sys.stdout.flush()

    def unravel_estimates(self, est_dict):

        pecmy = policies.policies(self.data, self.params, self.ROWname, self.bootstrap_id)

        xlhvt_star = np.genfromtxt(self.setup.xlhvt_star_path, delimiter=",")
        ge_x_star = pecmy.rewrap_xlhvt(xlhvt_star)["ge_x"]
        tau_star = pecmy.ecmy.rewrap_ge_dict(ge_x_star)["tau_hat"] * pecmy.ecmy.tau

        # X_star = pecmy.ecmy.rewrap_ge_dict(ge_x_star)["X_hat"] * pecmy.ecmy.Xcif
        # np.savetxt(self.setup.estimates_path + "X_star.csv", X_star, delimiter=",")

        v_star = pecmy.rewrap_xlhvt(xlhvt_star)["v"]
        theta_x_star = pecmy.rewrap_xlhvt(xlhvt_star)["theta"]
        theta_dict_star = pecmy.rewrap_theta(theta_x_star)
        for i in theta_dict_star.keys():
            np.savetxt(self.setup.estimates_path + i + ".csv", np.array([theta_dict_star[i]]), delimiter=",")
            if i in est_dict.keys():
                est_dict[i].append(theta_dict_star[i])
        np.savetxt(self.setup.estimates_path + "v.csv", v_star, delimiter=",")
        est_dict["v"].append(v_star)

        G_star = pecmy.G_hat(ge_x_star, v_star, 0, all=True)
        rcv_eq = pecmy.rcv_ft(ge_x_star, v_star)
        np.fill_diagonal(rcv_eq, 0)
        np.savetxt(self.setup.estimates_path + "rcv_eq.csv", rcv_eq, delimiter=",")
        est_dict["rcv"].append(rcv_eq.ravel())

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

        np.savetxt(self.setup.estimates_path + "peace_probs.csv", peace_prob_mat, delimiter=",")
        est_dict["peace_probs"].append(peace_prob_mat.ravel())

    def compute_counterfactual(self, v_star, theta_x_star, m):

        pecmy = policies.policies(self.data, self.params, self.ROWname, self.bootstrap_id)
        xlhvt_prime, obj, status = pecmy.estimator(v_star, theta_x_star, m, nash_eq=True)

        if status == 0:

            print(xlhvt_prime)
            print(obj)
            print(status)

            # np.savetxt(xlhvt_prime_path, x_path_base + "x.csv", delimiter=",")

        return(xlhvt_prime)

# bootstrap: https://github.com/brendancooley/tpsp/blob/f45862d123edce4ecaa026d7fa3947e3dc11cfb6/01_code/convexity_test.py
