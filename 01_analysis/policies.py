import autograd as ag
import autograd.numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import statsmodels.api as sm
import economy
import csv
import helpers_tpsp as hp
import time
import os
import copy
import multiprocessing as mp
import ipyopt

class policies:

    def __init__(self, data, params, ROWname, results_path):
        """

        Parameters
        ----------
        data : type
            Description of parameter `data`.
        params : type
            Description of parameter `params`.
        b : type
            Description of parameter `b`.
        rcv : string
            Path to regime change value matrix

        Returns
        -------
        type
            Description of returned object.

        """

        # Setup economy
        self.ecmy = economy.economy(data, params)
        self.N = self.ecmy.N
        self.ids = np.arange(self.N)

        self.ROW_id = np.where(data["ccodes"]==ROWname)[0][0]

        # purge deficits
        self.ecmy.purgeD()

        # enforce positive tariffs
        tau_hat_pos = np.ones_like(self.ecmy.tau)
        tau_hat_pos[self.ecmy.tau < 1] = 1 / self.ecmy.tau[self.ecmy.tau < 1]
        self.ecmy.update_ecmy(tau_hat_pos, np.ones(self.N))

        self.W = data["W"]  # distances
        np.fill_diagonal(self.W, 0)
        self.M = data["M"]  # milex

        self.m = self.M / np.ones((self.N, self.N))
        self.m = self.m.T
        self.m[self.ROW_id,:] = 0
        self.m[:,self.ROW_id] = 0
        self.m[self.ROW_id,self.ROW_id] = 1

        self.tauMin = 1  # enforce lower bound on policies
        self.tauMax = 15
        self.tau_nft = 1.25  # where to begin search for best response

        self.max_iter_ipopt = 100000

        self.hhat_len = self.N**2+4*self.N  # X, d, P, w, r, E
        self.Dhat_len = self.N
        self.tauj_len = self.N**2-self.N
        # self.lambda_i_len = self.hhat_len + self.tauj_len + 1 + self.N + (self.N - 1)  # ge vars, other policies, tau_ii, deficits, mil constraints
        # self.lambda_i_len = self.hhat_len + 1 + (self.N - 1)
        self.lambda_i_x_len = self.hhat_len # one is own policy (redundant?)
        self.lambda_i_len = self.lambda_i_x_len + self.N
        # self.lambda_i_len_td = self.lambda_i_len + self.N ** 2 - self.N # add constraints on others' policies

        self.x_len = self.ecmy.ge_x_len
        self.xlvt_len = self.x_len + self.lambda_i_len * self.N + self.N + 4

        self.g_len = self.hhat_len + (self.hhat_len + self.N - 1)*self.N + self.N**2 + self.N**2  # ge_diffs, Lzeros (own policies N-1), war_diffs mat, comp_slack mat

        self.chi_min = 1.0e-10
        self.wv_min = -1.0e4

        ge_x_ft_path = results_path + "ge_x_ft.csv"
        if not os.path.isfile(ge_x_ft_path):
            self.ge_x_ft = np.zeros((self.N, self.x_len))
            for i in range(self.N):
                print(str(i) + "'s free trade vector")
                ge_x_ft_i = self.ft_sv(i, np.ones(self.x_len))
                self.ge_x_ft[i, ] = ge_x_ft_i
            np.savetxt(ge_x_ft_path, self.ge_x_ft, delimiter=",")
        else:
            self.ge_x_ft = np.genfromtxt(ge_x_ft_path, delimiter=",")

        self.tick = 0

    def G_hat(self, x, v, id, sign=1, all=False):
        """Calculate changes in government welfare given ge inputs and outputs

        Parameters
        ----------
        ge_x : vector (TODO: documentation for arbitrary input vector)
            1d numpy array storing flattened ge inputs and outputs. See function for order of values.
        affinity : matrix
            N times N matrix of affinity shocks
        ids : vector
            ids of governments for which to return welfare changes. Defaults to all.
        sign : scalar
            Scales output. Use to convert max problems to min.

        Returns
        -------
        vector
            G changes for selected governments

        """

        # if affinity is None:
        #     affinity = np.zeros((self.N, self.N))

        ge_dict = self.ecmy.rewrap_ge_dict(x)
        Uhat = self.ecmy.U_hat(ge_dict)
        # Ghat = Uhat ** (1 - b) * ge_dict["r_hat"] ** b
        Ghat = Uhat * self.R_hat(ge_dict, v)

        # Ghat_a = affinity * Ghat
        # Ghat_out = Ghat + np.sum(Ghat_a, axis=1)
        if all == False:
            return(Ghat[id]*sign)
        else:
            return(Ghat*sign)

    def G_hat_grad(self, x, v, id, sign):
        G_hat_grad_f = ag.grad(self.G_hat)
        return(G_hat_grad_f(x, v, id, sign))

    def r_v(self, ge_dict, v):

        v_mat = np.array([v])
        tau_mv = self.ecmy.tau - np.tile(v_mat.transpose(), (1, self.N))
        tau_mv = tau_mv - np.diag(np.diag(tau_mv))
        # tau_mv[tau_mv < 0] = 0
        # tau_mv = np.clip(tau_mv, 0, np.inf)
        r = np.sum(tau_mv * self.ecmy.Xcif, axis=1)

        return(r)

    def R_hat(self, ge_dict, v):

        v_mat = np.array([v])
        r = self.r_v(ge_dict, v)

        tau_prime = ge_dict["tau_hat"] * self.ecmy.tau
        tau_prime_mv = tau_prime - np.tile(v_mat.transpose(), (1, self.N))
        tau_prime_mv = tau_prime_mv - np.diag(np.diag(tau_prime_mv))
        # tau_prime_mv = np.clip(tau_prime_mv, 0, np.inf)
        X_prime = ge_dict["X_hat"] * self.ecmy.Xcif
        r_prime = np.sum(tau_prime_mv * X_prime, axis=1)

        r_hat = r_prime / r

        return(r_hat)

    def rewrap_xlvt(self, xlvt):

        xlvt_dict = dict()
        xlvt_dict["ge_x"] = xlvt[0:self.x_len]
        xlvt_dict["lbda"] = xlvt[self.x_len:self.x_len+self.lambda_i_len*self.N]
        xlvt_dict["v"] = xlvt[self.x_len+self.lambda_i_len*self.N:self.x_len+self.lambda_i_len*self.N+self.N]
        xlvt_dict["theta"] = xlvt[self.x_len+self.lambda_i_len*self.N+self.N:]

        return(xlvt_dict)

    def unwrap_xlvt(self, xlvt_dict):

        xlvt = []
        xlvt.extend(xlvt_dict["ge_x"])
        xlvt.extend(xlvt_dict["lbda"])
        xlvt.extend(xlvt_dict["v"])
        xlvt.extend(xlvt_dict["theta"])

        return(np.array(xlvt))

    def rewrap_theta(self, theta_x):

        theta_dict = dict()
        theta_dict["c_hat"] = theta_x[0]
        theta_dict["gamma"] = theta_x[1]
        theta_dict["alpha0"] = theta_x[2]
        theta_dict["alpha1"] = theta_x[3]

        return(theta_dict)

    def unwrap_theta(self, theta_dict):

        theta_x = []
        theta_x.extend(np.array([theta_dict["c_hat"]]))
        theta_x.extend(np.array([theta_dict["gamma"]]))
        theta_x.extend(np.array([theta_dict["alpha0"]]))
        theta_x.extend(np.array([theta_dict["alpha1"]]))

        return(np.array(theta_x))

    def loss(self, xlvt):

        ge_x = self.rewrap_xlvt(xlvt)["ge_x"]
        v = self.rewrap_xlvt(xlvt)["v"]
        theta_x = self.rewrap_xlvt(xlvt)["theta"]

        self.tick += 1
        if self.tick % 25 == 0:
            print("ge_dict:")
            print(self.ecmy.rewrap_ge_dict(ge_x))

            print("v:")
            print(v)

            print("theta_dict:")
            print(self.rewrap_theta(theta_x))

            print("G_hat:")
            print(self.G_hat(ge_x, v, 0, all=True))

            print("R_hat:")
            print(self.R_hat(self.ecmy.rewrap_ge_dict(ge_x), v))


        tau_hat = self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"]
        tau_star = tau_hat * self.ecmy.tau

        tau_diffs = tau_star - self.ecmy.tau
        loss = np.sum(tau_diffs**2)

        # loss = 0
        # for i in range(self.N):
        #     loss += self.loss_tau(tau_hat[i, ], i)

        return(loss)

    def loss_tau(self, tau_i, id, weights=None):

        if weights is None:
            weights = np.ones(self.N)
        weights = weights / np.sum(weights)  # normalize

        tau_star = self.ecmy.tau[id, ] * tau_i
        out = np.sum((self.ecmy.tau[id, ] - tau_star) ** 2 * weights)

        return(out)

    def loss_grad(self, xlvt, out):

        loss_grad_f = ag.grad(self.loss)
        out[()] = loss_grad_f(xlvt)

        return(out)

    def geq_diffs_xlvt(self, xlvt):
        ge_x = self.rewrap_xlvt(xlvt)["ge_x"]
        return(self.ecmy.geq_diffs(ge_x))

    def Lzeros_i_xlvt(self, xlvt, id):

        xlvt_dict = self.rewrap_xlvt(xlvt)
        ge_x = xlvt_dict["ge_x"]
        lbda = np.reshape(xlvt_dict["lbda"], (self.N, self.lambda_i_len))
        lbda_i = lbda[id, ]
        v = xlvt_dict["v"]
        theta_x = xlvt_dict["theta"]
        theta_dict = self.rewrap_theta(theta_x)

        wv = self.war_vals(v, self.m, theta_dict)

        Lzeros_i = self.Lzeros_i(np.concatenate((ge_x, lbda_i)), id, v, wv[:,id])

        return(Lzeros_i)

    def war_diffs_xlvt(self, xlvt, id):

        xlvt_dict = self.rewrap_xlvt(xlvt)
        ge_x = xlvt_dict["ge_x"]
        v = xlvt_dict["v"]
        theta_x = xlvt_dict["theta"]
        theta_dict = self.rewrap_theta(theta_x)

        wv = self.war_vals(v, self.m, theta_dict)

        war_diffs_i = self.war_diffs(ge_x, v, wv[:,id], id)

        return(war_diffs_i)

    def comp_slack_xlvt(self, xlvt, id):

        xlvt_dict = self.rewrap_xlvt(xlvt)
        ge_x = xlvt_dict["ge_x"]
        lbda = np.reshape(xlvt_dict["lbda"], (self.N, self.lambda_i_len))
        lbda_i = lbda[id, ]
        v = xlvt_dict["v"]
        theta_x = xlvt_dict["theta"]
        theta_dict = self.rewrap_theta(theta_x)

        wv = self.war_vals(v, self.m, theta_dict)

        war_diffs_i = self.war_diffs(ge_x, v, wv[:,id], id)

        lbda_i_chi = self.rewrap_lbda_i(lbda_i)["chi_i"]

        comp_slack_i = war_diffs_i * lbda_i_chi

        return(comp_slack_i)

    def estimator_cons(self, xlvt, out):

        # geq constraints
        geq_diffs = self.geq_diffs_xlvt(xlvt)

        # Lagrange gradient
        Lzeros = []
        war_diffs = []
        comp_slack = []
        for i in range(self.N):
            Lzeros_i = self.Lzeros_i_xlvt(xlvt, i)
            Lzeros.extend(Lzeros_i)
            war_diffs_i = self.war_diffs_xlvt(xlvt, i)
            war_diffs.extend(war_diffs_i)
            comp_slack_i = self.comp_slack_xlvt(xlvt, i)
            comp_slack.extend(comp_slack_i)

        out[()] = np.concatenate((geq_diffs, Lzeros, war_diffs, comp_slack), axis=None)

        return(out)

    def estimator_cons_hess(self, xlvt):

        # geq constraints
        geq_diffs = self.geq_diffs_xlvt(xlvt)

        # Lagrange gradient
        Lzeros = []
        war_diffs = []
        comp_slack = []
        for i in range(self.N):
            Lzeros_i = self.Lzeros_i_xlvt(xlvt, i)
            Lzeros.extend(Lzeros_i)
            war_diffs_i = self.war_diffs_xlvt(xlvt, i)
            war_diffs.extend(war_diffs_i)
            comp_slack_i = self.comp_slack_xlvt(xlvt, i)
            comp_slack.extend(comp_slack_i)

        all = []
        all.extend(geq_diffs)
        all.extend(Lzeros)
        all.extend(war_diffs)
        all.extend(comp_slack)
        out = np.array(all)

        return(out)

    def estimator_cons_jac(self, xlvt, g_sparsity_bin):

        geq_diffs_jac_f = ag.jacobian(self.geq_diffs_xlvt)
        geq_diffs_jac = geq_diffs_jac_f(xlvt)

        Lzeros_i_jac_f = ag.jacobian(self.Lzeros_i_xlvt)
        war_diffs_i_jac_f = ag.jacobian(self.war_diffs_xlvt)
        comp_slack_i_jac_f = ag.jacobian(self.comp_slack_xlvt)

        Lzeros_jac_flat = []
        war_diffs_jac_flat = []
        comp_slack_flat = []
        for i in range(self.N):
            Lzeros_i_jac = Lzeros_i_jac_f(xlvt, i)
            Lzeros_jac_flat.extend(Lzeros_i_jac.ravel())
            war_diffs_i_jac = war_diffs_i_jac_f(xlvt, i)
            war_diffs_jac_flat.extend(war_diffs_i_jac.ravel())
            comp_slack_i_jac = comp_slack_i_jac_f(xlvt, i)
            comp_slack_flat.extend(comp_slack_i_jac.ravel())

        out_full = np.concatenate((geq_diffs_jac.ravel(), Lzeros_jac_flat, war_diffs_jac_flat, comp_slack_flat), axis=None)
        out = out_full[g_sparsity_bin]

        return(out)

    def estimator_cons_jac_wrap(self, g_sparsity_bin):
        def f(x, out):
            out[()] = self.estimator_cons_jac(x, g_sparsity_bin)
            return(out)
        return(f)

    def estimator_lgrg(self, xlvt, lagrange, obj_factor):

        loss = self.loss(xlvt)
        cons = self.estimator_cons_hess(xlvt)

        out = obj_factor * loss + np.sum(lagrange * cons)

        return(out)

    def estimator_lgrg_hess(self, xlvt, lagrange, obj_factor, out):

        lgrg_hess_f = ag.hessian(self.estimator_lgrg)
        lgrg_hess_mat = lgrg_hess_f(xlvt, lagrange, obj_factor)
        out[()] = lgrg_hess_mat.ravel()

        return(out)

    def g_sparsity_bin(self, xlvt_sv):

        jac_flat = self.estimator_cons_jac(xlvt_sv, np.repeat(True, self.xlvt_len*self.g_len))
        out = jac_flat != 0

        return(out)

    def g_sparsity_idx(self, g_sparsity_bin):

        jac_mat = np.reshape(g_sparsity_bin, (self.g_len, self.xlvt_len))
        print(np.sum(jac_mat==True) / (self.xlvt_len * self.g_len))
        out = np.argwhere(jac_mat==True)

        return(out)

    def estimator_bounds(self, bound="lower", nash_eq=False, theta_x=None, v=None):

        x_L = np.repeat(-np.inf, self.xlvt_len)
        x_U = np.repeat(np.inf, self.xlvt_len)

        tau_hat_lb = np.zeros((self.N, self.N))
        tau_hat_ub = np.max(self.ecmy.tau) / self.ecmy.tau
        np.fill_diagonal(tau_hat_lb, 1.)
        np.fill_diagonal(tau_hat_ub, 1.)

        x_L[0:self.x_len] = 0.
        x_L[0:self.N**2] = tau_hat_lb.ravel()
        x_U[0:self.N**2] = tau_hat_ub.ravel()
        x_L[self.N**2:self.N**2+self.N] = 1.
        x_U[self.N**2:self.N**2+self.N] = 1. # deficits

        lbda_i_bound_dict = dict()
        lbda_i_bound_dict["h_hat"] = np.repeat(-np.inf, self.hhat_len)
        lbda_i_bound_dict["chi_i"] = np.repeat(0., self.N)
        lbda_i_bound = self.unwrap_lbda_i(lbda_i_bound_dict)

        lbda_bound = np.tile(lbda_i_bound, self.N)

        x_L[self.x_len:self.x_len+self.lambda_i_len*self.N] = lbda_bound  # mil constraint multipliers

        if nash_eq == False:
            x_L[self.x_len+self.lambda_i_len*self.N:self.x_len+self.lambda_i_len*self.N+self.N] = 1 # vs
            x_U[self.x_len+self.lambda_i_len*self.N:self.x_len+self.lambda_i_len*self.N+self.N] = np.max(self.ecmy.tau) # vs
            x_L[self.x_len+self.lambda_i_len*self.N+self.N] = 0  # c_hat lower
            x_L[self.x_len+self.lambda_i_len*self.N+self.N+1] = 0  # gamma lower
            # x_L[self.x_len+self.lambda_i_len*self.N+self.N+2] = 0  # alpha0 lower
            # x_L[self.x_len+self.lambda_i_len*self.N+self.N+3] = 0  # alpha1 lower
            # x_L[self.x_len+self.lambda_i_len*self.N+self.N+1] = 1
            # x_U[self.x_len+self.lambda_i_len*self.N+self.N+1] = 1  # fix gamma at 1
            # x_L[self.x_len+self.lambda_i_len*self.N+self.N] = .25
            # x_U[self.x_len+self.lambda_i_len*self.N+self.N] = .25  # fix c_hat
            # x_L[self.x_len+self.lambda_i_len*self.N+self.N+1] = 0  # alpha lower
            # x_L[self.x_len+self.lambda_i_len*self.N+self.N+1] = 0
            # x_U[self.x_len+self.lambda_i_len*self.N+self.N+1] = 0  # fix alpha
            # x_L[self.x_len+self.lambda_i_len*self.N+self.N+2] = 0  # gamma lower
        else:
            theta_dict = self.rewrap_theta(theta_x)
            x_L[self.x_len+self.lambda_i_len*self.N:self.x_len+self.lambda_i_len*self.N+self.N] = v
            x_U[self.x_len+self.lambda_i_len*self.N:self.x_len+self.lambda_i_len*self.N+self.N] = v
            x_L[self.x_len+self.lambda_i_len*self.N+self.N] = theta_dict["c_hat"]  # c_hat
            x_U[self.x_len+self.lambda_i_len*self.N+self.N] = theta_dict["c_hat"]
            x_L[self.x_len+self.lambda_i_len*self.N+self.N+1] = theta_dict["gamma"]  # gamma
            x_U[self.x_len+self.lambda_i_len*self.N+self.N+1] = theta_dict["gamma"]
            x_L[self.x_len+self.lambda_i_len*self.N+self.N+2] = theta_dict["alpha0"]
            x_U[self.x_len+self.lambda_i_len*self.N+self.N+2] = theta_dict["alpha0"]
            x_L[self.x_len+self.lambda_i_len*self.N+self.N+3] = theta_dict["alpha1"]
            x_U[self.x_len+self.lambda_i_len*self.N+self.N+3] = theta_dict["alpha1"]

        if bound == "lower":
            return(x_L)
        else:
            return(x_U)

    def apply_new(_X):
        return(True)

    def estimator(self, v_sv, theta_x_sv, nash_eq=False):

        # if nash_eq = True fix theta vals at those in theta_dict_sv and compute equilibrium
        x_len = self.xlvt_len

        wd_g = np.repeat(np.inf, self.N**2)
        g_upper = np.zeros(self.g_len)
        g_upper[self.hhat_len + (self.hhat_len + self.N - 1)*self.N:self.hhat_len + (self.hhat_len + self.N - 1)*self.N+self.N**2] = wd_g

        xlvt_sv_dc = np.concatenate((np.ones(self.x_len), np.repeat(.01, self.lambda_i_len*self.N), v_sv, theta_x_sv))  # NOTE: for derivative checker, we will use these to calculate Jacobian sparsity
        xlvt_sv = np.concatenate((np.ones(self.x_len), np.zeros(self.lambda_i_len*self.N), v_sv, theta_x_sv))

        # Search entire Jacobian
        g_sparsity_indices_a = np.array(np.meshgrid(range(self.g_len), range(x_len))).T.reshape(-1,2)
        g_sparsity_indices = (g_sparsity_indices_a[:,0], g_sparsity_indices_a[:,1])
        g_sparsity_bin = np.repeat(True, self.g_len*self.xlvt_len)

        # Sparse Jacobian
        # NOTE: starting values sometimes induce sparsity for elements that have positive derivatives for some parameters. But problem seems to go away if we make wv_min low enough
            # attempting both versions of sparsity on mini problem
            # doit results: Sparse (TODO: need to debug...runs forever for some reason)
            # python: full
            # (holding gamma fixed at 1)
        # g_sparsity_bin = self.g_sparsity_bin(xlvt_sv)
        # g_sparsity_indices_a = self.g_sparsity_idx(g_sparsity_bin)
        # g_sparsity_indices = (g_sparsity_indices_a[:,0], g_sparsity_indices_a[:,1])

        # NOTE: ipopt requires Hessian of *Lagrangian*, see hs071.py
        h_sparsity_indices_a = np.array(np.meshgrid(range(self.xlvt_len), range(self.xlvt_len))).T.reshape(-1,2)
        h_sparsity_indices = (h_sparsity_indices_a[:,0], h_sparsity_indices_a[:,1])

        if nash_eq == False:
            b_L = self.estimator_bounds("lower")
            b_U = self.estimator_bounds("upper")
        else:
            b_L = self.estimator_bounds("lower", True, theta_x_sv, v_sv)
            b_U = self.estimator_bounds("upper", True, theta_x_sv, v_sv)

        if nash_eq == False:
            # problem = ipyopt.Problem(self.xlvt_len, b_L, b_U, self.g_len, np.zeros(self.g_len), g_upper, g_sparsity_indices, h_sparsity_indices, self.loss, self.loss_grad, self.estimator_cons, self.estimator_cons_jac_wrap(g_sparsity_bin), self.estimator_lgrg_hess, self.apply_new)
            problem = ipyopt.Problem(self.xlvt_len, b_L, b_U, self.g_len, np.zeros(self.g_len), g_upper, g_sparsity_indices, h_sparsity_indices, self.loss, self.loss_grad, self.estimator_cons, self.estimator_cons_jac_wrap(g_sparsity_bin))
            # problem.set(print_level=5, nlp_scaling_method="none", fixed_variable_treatment='make_parameter', max_iter=self.max_iter_ipopt, mu_strategy="adaptive")
            problem.set(print_level=5, fixed_variable_treatment='make_parameter', max_iter=self.max_iter_ipopt, mu_strategy="adaptive")
            # problem.set(print_level=5, fixed_variable_treatment='make_parameter', max_iter=self.max_iter_ipopt, derivative_test="first-order", point_perturbation_radius=0.)
        else:
            # ge_x_sv = self.v_sv_all(v_sv)
            # xlvt_sv = np.concatenate((ge_x_sv, np.zeros(self.lambda_i_len*self.N), v_sv, theta_x_sv))
            problem = ipyopt.Problem(self.xlvt_len, b_L, b_U, self.g_len, np.zeros(self.g_len), g_upper, g_sparsity_indices, h_sparsity_indices, self.dummy, self.dummy_grad, self.estimator_cons, self.estimator_cons_jac)
            problem.set(print_level=5, nlp_scaling_method="none", fixed_variable_treatment='make_parameter')
        print("solving...")
        _x, obj, status = problem.solve(xlvt_sv)

        return(_x, obj, status)

    def unwrap_lbda_i(self, lambda_dict_i):
        """Convert dictionary of multipliers for gov i into vector for Lagrangians

        Parameters
        ----------
        lambda_dict_i : dict
            Nested dictionary storing multipliers for constrained maximization problems

        Returns
        -------
        vector
            vector length of number of ge and tau constraints

        """

        x = []
        x.extend(lambda_dict_i["h_hat"])
        x.extend(lambda_dict_i["chi_i"])

        return(np.array(x))

    def rewrap_lbda_i(self, x):
        """Return dictionary of Lagrange multipliers from vector of multipliers for given gov id

        Parameters
        ----------
        x : array
            vector length of number of ge and tau constraints for government i

        Returns
        -------
        dict
            Updated nested dictionary

        """

        lambda_dict_i = dict()
        lambda_dict_i["h_hat"] = x[0:self.hhat_len]  # ge vars
        lambda_dict_i["chi_i"] = x[self.hhat_len:]  # mil constraints, threats against i

        return(lambda_dict_i)

    def Lagrange_i_x(self, ge_x, lambda_i_x, id, v, wv):
        """Short summary.

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs.
        tau_hat : matrix
            N times N matrix of initial taus.
        war_vals : vector
            Length N minus one vector of war values for each non-id country in war against id
        lambda_x : vector
            multipliers for gov id
        id : int
            government choosing policy

        Returns
        -------
        scalar
            Value for Lagrangian

        """

        lambda_dict_i = self.rewrap_lbda_i(lambda_i_x)
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)

        G_hat_i = self.G_hat(ge_x, v, id, sign=1)
        geq_diffs = self.ecmy.geq_diffs(ge_x)
        war_diffs = self.war_diffs(ge_x, v, wv, id)

        # wdz = np.where(war_diffs > 0, 0, war_diffs)
        # wdz = -1 * wdz
        wd = -1 * war_diffs

        L_i = G_hat_i - np.dot(lambda_dict_i["h_hat"], geq_diffs) - np.dot(lambda_dict_i["chi_i"], wd)

        return(L_i)

    def L_grad_i_ind(self, id):

        ind_dict = dict()
        ind_dict["tau_hat"] = np.reshape(np.repeat(False, self.N**2), (self.N, self.N))
        ind_dict["tau_hat"][id, ] = True
        np.fill_diagonal(ind_dict["tau_hat"], False)
        ind_dict["D_hat"] = np.repeat(False, self.N)
        ind_dict["X_hat"] = np.reshape(np.repeat(True, self.N**2), (self.N, self.N))
        ind_dict["P_hat"] = np.repeat(True, self.N)
        ind_dict["w_hat"] = np.repeat(True, self.N)
        ind_dict["r_hat"] = np.repeat(True, self.N)
        ind_dict["E_hat"] = np.repeat(True, self.N)

        return(self.ecmy.unwrap_ge_dict(ind_dict))

    def Lzeros_i(self, ge_x_lbda_i_x, id, v, wv):
        """Short summary.

        Parameters
        ----------
        x : vector
            1d numpy array storing flattened ge_x values
        tau_hat : matrix
            N times N matrix of initial taus.
        war_vals : vector
            Length N minus one vector of war values for each non-id country in war against id
        id : int
            government choosing policy
        bound : "lower" or "upper"
            if "upper" multiply output vector by -1

        Returns
        -------
        vec
            1d array storing flattened gradient of Lagrangian and constraint diffs

        """

        ge_x = ge_x_lbda_i_x[0:self.x_len]
        lambda_i_x = ge_x_lbda_i_x[self.x_len:]

        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        L_grad_f = ag.grad(self.Lagrange_i_x)
        L_grad = L_grad_f(ge_x, lambda_i_x, id, v, wv)
        L_grad_i = L_grad[self.L_grad_i_ind(id)]

        out = []
        out.extend(L_grad_i)

        return(L_grad_i)

    def Lzeros_i_cons(self, ge_x_lbda_i_x, out, id, v, wv):

        ge_x = ge_x_lbda_i_x[0:self.x_len]
        lambda_i_x = ge_x_lbda_i_x[self.x_len:]

        geq_diffs = self.ecmy.geq_diffs(ge_x)
        Lzeros = self.Lzeros_i(ge_x_lbda_i_x, id, v, wv)
        war_diffs = self.war_diffs(ge_x, v, wv, id)
        comp_slack = war_diffs * self.rewrap_lbda_i(lambda_i_x)["chi_i"]

        out[()] = np.concatenate((geq_diffs, Lzeros, war_diffs, comp_slack))

        return(out)

    def Lzeros_i_cons_wrap(self, id, v, wv):
        def f(x, out):
            return(self.Lzeros_i_cons(x, out, id, v, wv))
        return(f)

    def geq_diffs_lbda(self, ge_x_lbda_i_x):
        ge_x = ge_x_lbda_i_x[0:self.x_len]
        return(self.ecmy.geq_diffs(ge_x))

    def war_diffs_lbda(self, ge_x_lbda_i_x, v, wv, id):
        ge_x = ge_x_lbda_i_x[0:self.x_len]
        return(self.war_diffs(ge_x, v, wv, id))

    def comp_slack_lbda(self, ge_x_lbda_i_x, v, wv, id):

        ge_x = ge_x_lbda_i_x[0:self.x_len]
        lambda_i_x = ge_x_lbda_i_x[self.x_len:]

        war_diffs = self.war_diffs(ge_x, v, wv, id)
        comp_slack = war_diffs * self.rewrap_lbda_i(lambda_i_x)["chi_i"]

        return(comp_slack)

    def Lzeros_i_cons_jac(self, ge_x_lbda_i_x, out, id, v, wv):

        geq_diffs_jac_f = ag.jacobian(self.geq_diffs_lbda)
        geq_diffs_jac_mat = geq_diffs_jac_f(ge_x_lbda_i_x)

        Lzero_diffs_jac_f = ag.jacobian(self.Lzeros_i)
        Lzero_jac_f_mat = Lzero_diffs_jac_f(ge_x_lbda_i_x, id, v, wv)

        war_diffs_jac_f = ag.jacobian(self.war_diffs_lbda)
        war_diffs_jac_mat = war_diffs_jac_f(ge_x_lbda_i_x, v, wv, id)

        comp_slack_jac_f = ag.jacobian(self.comp_slack_lbda)
        comp_slack_jac_mat = comp_slack_jac_f(ge_x_lbda_i_x, v, wv, id)

        out[()] = np.concatenate((geq_diffs_jac_mat.ravel(), Lzero_jac_f_mat.ravel(), war_diffs_jac_mat.ravel(), comp_slack_jac_mat.ravel()))

        return(out)

    def Lzeros_i_cons_jac_wrap(self, id, v, wv):
        def f(x, out):
            return(self.Lzeros_i_cons_jac(x, out, id, v, wv))
        return(f)

    def Lzeros_i_bounds(self, ge_x_sv, id, bound="lower"):

        tau_hat = self.ecmy.rewrap_ge_dict(ge_x_sv)["tau_hat"]

        x_L = np.concatenate((np.zeros(self.x_len), np.repeat(-np.inf, self.lambda_i_len)))
        x_U = np.repeat(np.inf, self.x_len+self.lambda_i_len)

        tau_L = 1. / self.ecmy.tau
        # tau_U = np.reshape(np.repeat(np.inf, self.N ** 2), (self.N, self.N))
        tau_U = (np.max(self.ecmy.tau, axis=1) / self.ecmy.tau.T).T
        np.fill_diagonal(tau_U, 1.)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    if i != id:
                        tau_L[i, j] = tau_hat[i, j]
                        tau_U[i, j] = tau_hat[i, j]
                    else:
                        tau_L[i, j] = 1 / self.ecmy.tau[i, j]
                else:
                    tau_L[i, j] = 1.
                    tau_U[i, j] = 1.


        x_L[0:self.N**2] = tau_L.ravel()
        x_U[0:self.N**2] = tau_U.ravel()

        # deficits
        x_L[self.N**2:self.N**2+self.N] = 1.
        x_U[self.N**2:self.N**2+self.N] = 1.

        x_L[-self.N:] = 0  # mil constraint multipliers

        if bound == "lower":
            return(x_L)
        else:
            return(x_U)

    def dummy(self, x):
        c = 1
        return(c)

    def dummy_grad(self, x, out):
        out[()] = np.zeros(len(x))
        return(out)

    def Lsolve_i_ipopt(self, id, v, wv):

        # verbose
        ipyopt.set_loglevel(ipyopt.LOGGING_DEBUG)

        ge_x0 = self.v_sv(id, np.ones(self.x_len), v)
        lbda_i0 = np.zeros(self.lambda_i_len)
        x0 = np.concatenate((ge_x0, lbda_i0))
        x_len = len(x0)

        g_len_i = self.hhat_len + (self.hhat_len + self.N - 1) + self.N + self.N  # ge constraints, gradient  war diffs, complementary slackness
        g_upper = np.zeros(g_len_i)
        g_upper[self.hhat_len + (self.hhat_len + self.N - 1):self.hhat_len + (self.hhat_len + self.N - 1)+self.N] = np.inf
        print(x_len)
        print(g_len_i)
        print(g_upper)

        g_sparsity_indices_a = np.array(np.meshgrid(range(g_len_i), range(x_len))).T.reshape(-1,2)
        g_sparsity_indices = (g_sparsity_indices_a[:,0], g_sparsity_indices_a[:,1])
        h_sparsity_indices_a = np.array(np.meshgrid(range(x_len), range(x_len))).T.reshape(-1,2)
        h_sparsity_indices = (h_sparsity_indices_a[:,0], h_sparsity_indices_a[:,1])

        problem = ipyopt.Problem(x_len, self.Lzeros_i_bounds(ge_x0, id, "lower"), self.Lzeros_i_bounds(ge_x0, id, "upper"), g_len_i, np.zeros(g_len_i), g_upper, g_sparsity_indices, h_sparsity_indices, self.dummy, self.dummy_grad, self.Lzeros_i_cons_wrap(id, v, wv), self.Lzeros_i_cons_jac_wrap(id, v, wv))

        problem.set(print_level=5, nlp_scaling_method="none", fixed_variable_treatment='make_parameter')
        print("solving...")
        x_lbda, obj, status = problem.solve(x0)

        return(x_lbda, obj, status)

        return(x_lbda)

    def war_vals(self, v, m, theta_dict):
        """Calculate war values (regime change value minus war costs)

        Parameters
        ----------
        b : vector
            length N vector of preference parameters
        m : matrix
            N times N matrix of military deployments.
        theta_dict : dict
            Dictionary storing military structural parameters
        id : int
            defending country id
        c_bar : float
            maximum adjusted war cost to return if chi_ji = 0

        Returns
        -------
        vector
            Matrix of war values for row id in war against col id, zeros on diagonal

        """

        chi = self.chi(m, theta_dict)
        chi = np.clip(chi, self.chi_min, 1)
        wc = theta_dict["c_hat"] / chi
        rcv_ft = self.rcv_ft(v)
        wv = rcv_ft - wc
        wv = np.clip(wv, self.wv_min, np.inf)

        return(wv)

    def rcv_ft(self, v):

        out = np.array([self.G_hat(self.ge_x_ft[i, ], v, 0, all=True) for i in range(self.N)])

        return(out.T)

    def Lzeros_jac(self, ge_x_lbda_i_x, v, tau_hat, war_vals, id, enforce_geq, bound):
        Lzeros_jac_f = ag.jacobian(self.Lzeros)
        return(Lzeros_jac_f(ge_x_lbda_i_x, v, tau_hat, war_vals, id, enforce_geq, bound))

    def Lzeros_cor(self, x_lbda, v, wv):

        ge_x = x_lbda[0:self.x_len]
        lbda = np.reshape(x_lbda[self.x_len:], (self.N, self.lambda_i_len))

        out = []
        for i in range(self.N):
            lbda_i = lbda[i, ]
            ge_x_lbda_i_x = np.concatenate((ge_x, lbda_i))
            out.extend(self.Lzeros(ge_x_lbda_i_x, v, self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"], wv[:,i], i, enforce_geq=False, bound="lower"))

        out.extend(self.ecmy.geq_diffs(ge_x))
        out.extend(self.ecmy.rewrap_ge_dict(ge_x)["D_hat"] - 1)

        return(np.array(out))

    def Lzeros_cor_jac(self, x_lbda, v, wv):
        Lzeros_cor_jac_f = ag.jacobian(self.Lzeros_cor)
        return(Lzeros_cor_jac_f(x_lbda, v, wv))

    def Lzeros_eq(self, v, wv):

        x_lbda_sv = np.zeros(self.x_len+self.N*self.lambda_i_len)
        x_lbda_sv[0:self.x_len] = 1
        out = opt.root(self.Lzeros_cor, x_lbda_sv, args=(v, wv, ), method="lm", jac=self.Lzeros_cor_jac)
        # out = opt.fsolve(self.Lzeros_cor, x_lbda_sv, args=(v, wv, ))

        return(out)

    def Lzeros_i_wrap(self, x_lbda_theta, m, id, enforce_geq=False, bound="lower"):

        ge_x = x_lbda_theta[0:self.x_len]
        lbda = np.reshape(x_lbda_theta[self.x_len:self.x_len+self.lambda_i_len*self.N], (self.N, self.lambda_i_len))
        theta = x_lbda_theta[self.x_len+self.lambda_i_len*self.N:]

        v = theta[0:self.N]
        theta_dict = dict()
        # TODO need to rewrite order
        # theta_dict["c_hat"] = theta[self.N]
        # theta_dict["alpha"] = theta[self.N+1]
        # theta_dict["gamma"] = theta[self.N+2]

        wv = self.war_vals(v, m, theta_dict, np.zeros((self.N, self.N)))

        ge_x_lbda_i_x = np.concatenate((ge_x, lbda[id, ]))
        # Lzero_i = self.Lzeros(ge_x_lbda_i_x, v, self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"], wv[:,id], id, enforce_geq=True)
        Lzero_i = self.Lzeros(ge_x_lbda_i_x, v, self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"], wv[:,id], id, enforce_geq=enforce_geq)

        if bound == "lower":
            return(-1*Lzero_i)
        else:
            return(Lzero_i)

    def Lzeros_i_wrap_jac(self, x_lbda_theta, m, id, enforce_geq, bound):
        Lzeros_i_wrap_jac_f = ag.jacobian(self.Lzeros_i_wrap)
        return(Lzeros_i_wrap_jac_f(x_lbda_theta, m, id, enforce_geq, bound))

    def Lzeros_min(self, v_init, theta_dict_init, mtd="SLSQP"):

        x_lbda_theta_sv = np.zeros(self.x_len+self.lambda_i_len*self.N+3+self.N)
        # x_lbda_theta_sv = np.random.normal(0, .1, size=self.x_len+self.lambda_i_len*self.N+3+self.N)
        x_lbda_theta_sv[0:self.x_len] = 1
        x_lbda_theta_sv[self.x_len+self.lambda_i_len*self.N:self.x_len+self.lambda_i_len*self.N+self.N] = v_init
        # TODO need to rewrite order
        # x_lbda_theta_sv[self.x_len+self.lambda_i_len*self.N+self.N] = theta_dict_init["c_hat"]
        # x_lbda_theta_sv[self.x_len+self.lambda_i_len*self.N+self.N+1] = theta_dict_init["alpha"]
        # x_lbda_theta_sv[self.x_len+self.lambda_i_len*self.N+self.N+2] = theta_dict_init["gamma"]

        m = self.M / np.ones((self.N, self.N))
        m = m.T

        # constrain deficits
        def con_d(ge_x, bound="lower"):
            if bound == "lower":
                con = ge_x[self.N**2:self.N**2+self.N] - 1
            else:
                con = 1 - ge_x[self.N**2:self.N**2+self.N]
            return(con)

        def con_d_grad(ge_x, bound="lower"):
            con_d_grad_f = ag.jacobian(con_d)
            return(con_d_grad_f(ge_x, bound))

        cons = []
        if mtd == "SLSQP":
            for i in range(self.N):
                cons.append({'type': 'ineq','fun': self.Lzeros_i_wrap, 'jac':self.Lzeros_i_wrap_jac, 'args':(m, i, True, "lower", )})
                cons.append({'type': 'ineq','fun': self.Lzeros_i_wrap, 'jac':self.Lzeros_i_wrap_jac, 'args':(m, i, True, "upper", )})
                # cons.append({'type': 'eq','fun': self.Lzeros_i_wrap, 'jac':self.Lzeros_i_wrap_jac, 'args':(m, i, False, "lower",)})

            cons.append({'type': 'eq', 'fun': self.ecmy.geq_diffs, 'jac': self.ecmy.geq_diffs_grad, 'args':("lower",)})
            # cons.append({'type': 'ineq', 'fun': self.ecmy.geq_diffs, 'jac': self.ecmy.geq_diffs_grad, 'args':("lower",)})
            # cons.append({'type': 'ineq', 'fun': self.ecmy.geq_diffs, 'jac': self.ecmy.geq_diffs_grad, 'args':("upper",)})

            cons.append({'type': 'eq', 'fun': con_d, 'jac': con_d_grad, 'args':("lower",)})
            # cons.append({'type': 'ineq', 'fun': con_d, 'jac': con_d_grad, 'args':("lower",)})
            # cons.append({'type': 'ineq', 'fun': con_d, 'jac': con_d_grad, 'args':("upper",)})

        def Lzeros_i_wrap_tc(m, i, bound):
            def f(x):
                return(self.Lzeros_i_wrap(x, m, i, bound))
            def f_jac(x):
                return(self.Lzeros_i_wrap_jac(x, m, i, bound))
            return(f, f_jac)

        if mtd == "trust-constr":
            for i in range(self.N):
                cons.append(opt.NonlinearConstraint(Lzeros_i_wrap_tc(m, i, "lower")[0], 0, 0, jac=Lzeros_i_wrap_tc(m, i, "lower")[1], hess=opt.BFGS()))

            cons.append(opt.NonlinearConstraint(self.ecmy.geq_diffs, 0, 0, jac=self.ecmy.geq_diffs_grad, hess=opt.BFGS()))
            cons.append(opt.NonlinearConstraint(con_d, 0, 0, jac=con_d_grad, hess=opt.BFGS()))


        bounds = []
        for i in range(self.N**2):
            bounds.append((1, None))
        for i in range(self.Dhat_len+self.hhat_len):
            bounds.append((.01, None))
        for i in range(self.lambda_i_len*self.N):
            bounds.append((None, None))
        for i in range(self.N):
            bounds.append((1, self.v_max[i]))
        for i in range(2):
            bounds.append((0, None))
        bounds.append((0, 2))  # gamma

        out = opt.minimize(self.Lzeros_loss, x_lbda_theta_sv, method=mtd, constraints=cons, bounds=bounds)
        # while out['success'] == False:
        #     print(out)
        #     print("iterating...")
        #     x = out['x'] + np.random.normal(0, .001, size=len(x_lbda_theta_sv))
        #     out = opt.minimize(self.Lzeros_loss, x, method=mtd, constraints=cons, bounds=bounds)

        return(out)

    def war_diffs(self, ge_x, v, war_vals, id):
        """Calculate difference between government id's utility at proposed vector ge_x versus war_value

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs.
        war_vals : vector
            Length N vector of war values for each non-id country in war against id
        id : int
            Gov id for which to calculate war constraints

        Returns
        -------
        vector
            Length N minus one vector of peace-war against id utility differences

        """

        G = self.G_hat(ge_x, v, 0, all=True)
        war_diffs = G - war_vals

        # turn to zero where negative
        # wdz = np.where(war_diffs < 0, 0, war_diffs)

        return(war_diffs)

    def chi(self, m, theta_dict):

        rhoM = self.rhoM(theta_dict)
        m_diag = np.diagonal(m)
        m_frac = m / m_diag

        chi_logit = rhoM * m_frac ** theta_dict["gamma"]

        chi = chi_logit / (1 + chi_logit)

        return(chi)

    def rhoM(self, theta_dict):
        """Calculate loss of strength gradient given alphas and distance matrix (W)

        Parameters
        ----------
        theta_dict : dict
            Dictionary storing military structural parameters

        Returns
        -------
        matrix
            N times N symmetric matrix loss of strength gradient

        """

        # rhoM = np.exp(-1 * (theta_dict["alpha"][0] + self.W * theta_dict["alpha"][1]) + epsilon)
        rhoM = np.exp(-1 * (theta_dict["alpha0"] + self.W * theta_dict["alpha1"]))
        # rhoM = np.clip(rhoM, 0, 1)

        return(rhoM)

    def Lsolve(self, v, m, theta_dict, id, epsilon=None, mtd="lm", enforce_geq=False):
        """Solves for zeros of Lagrange optimality conditions for id's policies, holding others' at values in tau_hat. If ft==True, algorithm begins searching at free trade values for gov id. Otherwise, tries "lm" and "hybr" methods recursively starting at tau_hat, before trying each starting at free trade.

        Parameters
        ----------
        tau_hat : matrix
            N times N matrix of policies
        m : matrix
            N times N matrix of military deployments.
        id : int
            Which government to solve for
        ft : bool
            Start id's values at free trade?
        mtd : "lm" or "hybr"
            Method for root finder

        Returns
        -------
        vector
            ge_x values at optimum

        """

        if epsilon is None:
            epsilon = np.zeros((self.N, self.N))

        ge_x_sv = self.v_sv(id, np.ones(self.x_len), v)
        ge_x_sv = np.ones(self.x_len)
        ge_dict_sv = self.ecmy.rewrap_ge_dict(ge_x_sv)

        lambda_i_x_sv = np.zeros(self.lambda_i_len)

        # calculate war values
        wv = self.war_vals(v, m, theta_dict, epsilon)
        wv_i = wv[:,id]
        # wv_i = np.zeros(self.N)

        # x = []
        # x.extend(ge_dict_sv["tau_hat"][id, ])
        # x.extend(ge_x_sv[self.N**2+self.N:self.x_len])
        # x.extend(lambda_i_x_sv)

        ge_x_lbda_i_x = np.concatenate((ge_x_sv, lambda_i_x_sv))

        # fct = .1  # NOTE: convergence of hybr and lm is sensitive to this value
        # out = opt.root(self.Lzeros_tixlbda, x0=np.array(x), method=mtd, args=(v, ge_dict_sv["tau_hat"], wv_i, id, True, ), options={"factor":fct})
        out = opt.root(self.Lzeros, x0=ge_x_lbda_i_x, method=mtd, jac=self.Lzeros_jac, args=(v, ge_dict_sv["tau_hat"], wv_i, id, enforce_geq, "lower", ), options={'ftol':1e-12, 'xtol':1e-12})
        # out = opt.root(self.Lzeros, x0=ge_x_lbda_i_x, method=mtd, args=(v, ge_dict_sv["tau_hat"], wv_i, id, enforce_geq, "lower", ), options={'ftol':1e-12, 'xtol':1e-12})
        if out['success'] == True:
            print("success:" + str(id))
            print(out)
            return(out['x'])
        # else:
        #     print("recursing...")
        #     if mtd == "lm":  # first try hybr
        #         return(self.Lsolve(tau_hat, v, m, theta_dict, id, mtd="hybr"))
        #     else:  # otherwise start from free trade
        #         return(self.Lsolve(tau_hat, v, m, theta_dict, id, id, mtd="lm"))

    def constraints_tau(self, ge_dict, tau_free, wv_i, v, ge=True, deficits=True, mil=False):
        """Constructs list of constraints for policy br.

        Parameters
        ----------
        ge_dict : dictionary
            Stores GE inputs and outputs.
        tau_free : integer
            Which governments policies are choice variables?
        m : matrix
            N times N matrix of military deployments.
        ge : bool
            Include GE constraints (for mpecs)?
        deficits : bool
            Include deficit constraints?

        Returns
        -------
        list
            List of constraints for scipy SQSLP optimizers.

        """

        # if m is None:
        #     m = np.diag(self.M)

        # constrain policies
        # constrain row i, column j of trade policy matrix
        def con_tau(i, j, bound="lower", bv=1):
            def f(x):
                if bound == "lower":
                    con = x[i*self.N + j] - bv
                else:
                    con = bv - x[i*self.N + j]
                return(con)
            def f_grad(x):
                f_grad_f = ag.grad(f)
                return(f_grad_f(x))
            return(f, f_grad)

        # constrain deficits
        def con_d(ge_x, bound="lower"):
            if bound == "lower":
                con = ge_x[self.N**2:self.N**2+self.N] - 1
            else:
                con = 1 - ge_x[self.N**2:self.N**2+self.N]
            return(con)

        def con_d_grad(ge_x, bound):
            con_d_grad_f = ag.jacobian(con_d)
            return(con_d_grad_f(ge_x, bound))

        # build constraints
        cons = []

        # policies
        for i in np.arange(0, self.N):
            for j in np.arange(0, self.N):
                if i != tau_free:
                    cons.append({'type': 'eq','fun': con_tau(i, j, bound="lower", bv=ge_dict["tau_hat"][i, j])[0], 'jac':con_tau(i, j, bound="lower", bv=ge_dict["tau_hat"][i, j])[1]})
                    # cons.append({'type': 'ineq','fun': con_tau(i, j, bound="upper", bv=ge_dict["tau_hat"][i, j])[0], 'jac':con_tau(i, j, bound="upper", bv=ge_dict["tau_hat"][i, j])[1]})
                else:
                    if i == j:
                        cons.append({'type': 'eq','fun': con_tau(i, j, bound="lower", bv=1)[0], 'jac':con_tau(i, j, bound="lower", bv=ge_dict["tau_hat"][i, j])[1]})
                        # cons.append({'type': 'ineq','fun': con_tau(i, j, bound="upper", bv=1)[0], 'jac':con_tau(i, j, bound="upper", bv=ge_dict["tau_hat"][i, j])[1]})
                    else:
                        cons.append({'type': 'ineq','fun': con_tau(i, j, bound="lower", bv=0)[0], 'jac':con_tau(i, j, bound="lower", bv=ge_dict["tau_hat"][i, j])[1]})

        # deficits
        if deficits == True:
            cons.append({'type': 'eq', 'fun': con_d, 'jac': con_d_grad, 'args':("lower",)})
            # cons.append({'type': 'ineq', 'fun': con_d, 'jac': con_d_grad, 'args':("upper",)})

        # ge constraints
        if ge == True:
            cons.append({'type': 'eq', 'fun': self.ecmy.geq_diffs, 'jac': self.ecmy.geq_diffs_grad, 'args':("lower",)})
            # cons.append({'type': 'ineq', 'fun': self.ecmy.geq_diffs, 'jac': self.ecmy.geq_diffs_grad, 'args':("upper",)})

        # mil constraints
        if mil == True:
            for j in range(self.N):
                if j != tau_free:
                    cons.append({'type': 'ineq', 'fun': self.con_mil, 'jac':self.con_mil_grad, 'args':(tau_free, j, wv_i[j], v, )})

        return(cons)

    def con_mil(self, ge_x, i, j, wv_ji, v):
        """

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function for order of values.
        i : int
            Defending country id
        j : int
            Constraining country id
        m : matrix
            N times N matrix of military deployments.

        Returns
        -------
        float
            constraint satisfied at ge_x when this is positive

        """
        # G_j
        G_j = self.G_hat(ge_x, v, j)

        cons = G_j - wv_ji

        return(cons)

    def con_mil_grad(self, ge_x, i, j, wv_ji, v):
        con_mil_grad_f = ag.jacobian(self.con_mil)
        return(con_mil_grad_f(ge_x, i, j, wv_ji, v))

    def bounds(self):
        """Generate bounds for optimization algorithms. Currently requires tariffs be positive.

        Returns
        -------
        list
            List of tuples length self.ecmy.ge_x_len

        """

        bnds = []
        tauHatMin = self.tauMin / self.ecmy.tau
        tauHatMax = self.tauMax / self.ecmy.tau
        for i in range(self.N):
            for j in range(self.N):
                # bnds.append((tauHatMin[i,j], tauHatMax[i,j]))
                bnds.append((tauHatMin[i,j], None))
        for i in range(self.ecmy.ge_x_len-self.N**2):
            bnds.append((.01, None))  # positive other entries

        return(bnds)

    def G_hat_grad_ipyopt(self, ge_x, out, v, id):
        G_hat_grad_f = ag.grad(self.G_hat)
        out[0:len(out)] = G_hat_grad_f(ge_x, v, id, -1)

    def G_hat_grad_ipyopt_wrap(self, v, id):
        def f(ge_x, out):
            return(self.G_hat_grad_ipyopt(ge_x, out, v, id))
        return(f)

    def br_cons_ipyopt(self, ge_x, out, id, v, wv=None):

        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        geq_diffs = self.ecmy.geq_diffs(ge_x)
        if not wv is None:
            war_diffs = self.war_diffs(ge_x, v, wv, id)
            out[()] = np.concatenate((geq_diffs, war_diffs))
        else:
            out[()] = np.array(geq_diffs)

        return(out)

    def br_cons_ipyopt_wrap(self, id, v, wv):
        def f(x, out):
            return(self.br_cons_ipyopt(x, out, id, v, wv))
        return(f)

    def br_cons_ipyopt_jac(self, ge_x, out, id, v, wv):

        geq_jac_f = ag.jacobian(self.ecmy.geq_diffs)
        mat_geq = geq_jac_f(ge_x)

        if not wv is None:
            wd_jac_f = ag.jacobian(self.war_diffs)
            mat_wd = wd_jac_f(ge_x, v, wv, id)
            out[()] = np.concatenate((mat_geq.ravel(), mat_wd.ravel()))
        else:
            out[()] = mat_geq.ravel()
        return(out)

    def br_cons_ipyopt_jac_wrap(self, id, v, wv):
        def f(x, out):
            return(self.br_cons_ipyopt_jac(x, out, id, v, wv))
        return(f)

    def br_bounds_ipyopt(self, ge_x_sv, id, bound="lower"):

        tau_hat = self.ecmy.rewrap_ge_dict(ge_x_sv)["tau_hat"]

        # TODO: set lower bounds slightly above zero
        x_L = np.zeros(self.x_len)
        x_U = np.repeat(np.inf, self.x_len)

        tau_L = 1. / self.ecmy.tau
        # tau_U = np.reshape(np.repeat(np.inf, self.N ** 2), (self.N, self.N))
        tau_U = np.max(self.ecmy.tau) / self.ecmy.tau
        np.fill_diagonal(tau_U, 1.)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    if i != id:
                        tau_L[i, j] = tau_hat[i, j]
                        tau_U[i, j] = tau_hat[i, j]
                    else:
                        tau_L[i, j] = 1 / self.ecmy.tau[i, j]
                else:
                    tau_L[i, j] = 1.
                    tau_U[i, j] = 1.


        x_L[0:self.N**2] = tau_L.ravel()
        x_U[0:self.N**2] = tau_U.ravel()

        # deficits
        x_L[self.N**2:self.N**2+self.N] = 1.
        x_U[self.N**2:self.N**2+self.N] = 1.

        if bound == "lower":
            return(x_L)
        else:
            return(x_U)

    def G_hat_wrap(self, v, id, sign):
        def f(x):
            return(self.G_hat(x, v, id, sign=sign))
        return(f)

    def br_ipyopt(self, x0, v, id, wv=None):

        # verbose
        ipyopt.set_loglevel(ipyopt.LOGGING_DEBUG)

        print(x0)
        # g_len = self.x_len - (self.N - 1)
        geq_c_len = self.x_len - self.N**2 - self.N
        if not wv is None:
            g_len_br = self.x_len - self.N**2
            g_upper = np.zeros(self.x_len - self.N**2)
            g_upper[geq_c_len:] = np.inf
        else:
            g_len_br = geq_c_len

        g_sparsity_indices_a = np.array(np.meshgrid(range(g_len_br), range(self.x_len))).T.reshape(-1,2)
        g_sparsity_indices = (g_sparsity_indices_a[:,0], g_sparsity_indices_a[:,1])
        h_sparsity_indices_a = np.array(np.meshgrid(range(self.x_len), range(self.x_len))).T.reshape(-1,2)
        h_sparsity_indices = (h_sparsity_indices_a[:,0], h_sparsity_indices_a[:,1])

        x_L = self.br_bounds_ipyopt(x0, id, "lower")
        x_U = self.br_bounds_ipyopt(x0, id, "upper")

        problem = ipyopt.Problem(self.x_len, x_L, x_U, g_len_br, np.zeros(g_len_br), g_upper, g_sparsity_indices, h_sparsity_indices, self.G_hat_wrap(v, id, -1), self.G_hat_grad_ipyopt_wrap(v, id), self.br_cons_ipyopt_wrap(id, v, wv), self.br_cons_ipyopt_jac_wrap(id, v, wv))

        problem.set(print_level=5, nlp_scaling_method="none", fixed_variable_treatment='make_parameter')
        print("solving...")
        x, obj, status = problem.solve(x0)

        return(x)

    def br_cor_ipyopt(self, ge_x, wv, v):

        tau_hat = np.zeros((self.N, self.N))
        for i in range(self.N):
            print("solving:")
            print(i)
            x0 = self.v_sv(i, ge_x, v)
            # ge_x_i = self.br_ipyopt(ge_x, v, i, wv[:,i])
            print(ge_x_i)
            tau_hat[i, ] = self.ecmy.rewrap_ge_dict(ge_x_i)["tau_hat"][i, ]

        print("-----")
        print("fp iteration:")
        print(tau_hat)
        ge_out_dict = self.ecmy.geq_solve(tau_hat, np.ones(self.N))
        print(ge_out_dict)
        print("-----")

        return(self.ecmy.unwrap_ge_dict(ge_out_dict))

    def nash_eq_ipyopt(self, v, theta_x):

        wv = self.war_vals(v, self.m, self.rewrap_theta(theta_x))
        ge_x_sv = np.ones(self.x_len)

        out = opt.fixed_point(self.br_cor_ipyopt, ge_x_sv, args=(wv, v, ), method="iteration")

        return(out)


    def br(self, ge_x, v, wv_i, id, mil=True, method="SLSQP", affinity=None):
        """Calculate optimal policies for gov id, given others' policies in ge_x.

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function for order of values.
        b : vector
            N times 1 vector of preference parameters
        m : matrix
            N times N matrix of military deployments.
        wv_i : vector
            Length N vector of war values (0 for i)
        id : int
            id of government for which to calculate best response
        mpec : bool
            Calculate wrt mpec (true) or enforce equilibrium in calculation (false)

        Returns
        -------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs at best response values.

        """

        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        tau_hat_ft = 1 / self.ecmy.tau
        v = np.copy(v)

        # initialize starting values of ge_x to equilibrium
        # ge_dict = self.ecmy.geq_solve(ge_dict["tau_hat"], ge_dict["D_hat"])
        # ge_x = self.ecmy.unwrap_ge_dict(ge_dict)
        # NOTE: don't want to repeat this for every iteration of br

        mxit = 500
        # eps = 1.0e-10

        b_perturb = .01
        tau_perturb = .01

        cons = self.constraints_tau(ge_dict, id, wv_i, v, mil=mil)
        bnds = self.bounds()
        if affinity is None:
            thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, jac=self.G_hat_grad, args=(v, id, -1, ), method="SLSQP", options={"maxiter":mxit})
        else:
            thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, jac=self.G_hat_grad, args=(v, id, -1, ), method="SLSQP", options={"maxiter":mxit})

        thistar_dict = self.ecmy.rewrap_ge_dict(thistar['x'])
        taustar = thistar_dict["tau_hat"]*self.ecmy.tau

        # try new starting values if we don't converge
        while thistar['success'] == False or np.any(np.isnan(thistar['x'])) or np.any(thistar_dict["tau_hat"] < 0):
            print("br unsuccessful, iterating...")
            for j in range(self.N):
                if id != j:
                    # ge_dict["tau_hat"][id, j] = thistar_dict["tau_hat"][id, j] + np.random.normal(loc=0, scale=tau_perturb)
                    ge_dict["tau_hat"][id, j] += tau_perturb
            ge_dict = self.ecmy.geq_solve(ge_dict["tau_hat"], ge_dict["D_hat"])
            print(ge_dict)
            ge_x = self.ecmy.unwrap_ge_dict(ge_dict)
            # b[id] += b_perturb * np.random.choice([-1, 1]) # perturb preference value
            # print(b)
            thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, jac=self.G_hat_grad, args=(v, id, -1, ), method="SLSQP", options={"maxiter":mxit})
            thistar_dict = self.ecmy.rewrap_ge_dict(thistar['x'])
            print("taustar_out:")
            print(thistar_dict["tau_hat"]*self.ecmy.tau)

        # NOTE: sometimes extreme values due to difficulties in satisfying particular mil constraints
        # NOTE: one way to fix this is is to just force countries to impose free trade when they win wars, no manipulation
        # Also seems to happen when target countries are small, easy to make mistakes in finite difference differentiation?
        while np.any(taustar > self.tauMax):
            print("extreme tau values found, iterating...")
            print("taustar[id]: " + str(taustar[id, ]))
            for j in range(self.N):
                # if taustar[id, j] > self.tauMax:
                #     ge_dict["tau_hat"][id, j] = tau_hat_ft[id, j]
                # else:
                if id != j:
                    # ge_dict["tau_hat"][id, j] = thistar_dict["tau_hat"][id, j] + np.random.normal(loc=0, scale=tau_perturb)
                    ge_dict["tau_hat"][id, j] += tau_perturb
            ge_dict = self.ecmy.geq_solve(ge_dict["tau_hat"], ge_dict["D_hat"])
            print(ge_dict)
            # ge_x = self.ecmy.unwrap_ge_dict(ge_dict)
            # print(ge_dict["tau_hat"] * self.ecmy.tau)
            # ge_dict = self.ecmy.geq_solve(ge_dict["tau_hat"], ge_dict["D_hat"])
            # ge_x = self.ecmy.unwrap_ge_dict(ge_dict)
            # b[id] += b_perturb * np.random.choice([-1, 1]) # perturb preference value
            # ge_dict = self.ecmy.geq_solve(tau_hat_ft, ge_dict["D_hat"])
            # ge_dict["tau_hat"][id, ] += .1  # bump up starting taus
            # ge_dict["tau_hat"][id, id] = 1
            # ge_dict = self.ecmy.geq_solve(ge_dict["tau_hat"], ge_dict["D_hat"])
            ge_x = self.ecmy.unwrap_ge_dict(ge_dict)
            thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, jac=self.G_hat_grad, args=(v, id, -1, ), method="SLSQP", options={"maxiter":mxit})
            thistar_dict = self.ecmy.rewrap_ge_dict(thistar['x'])
            taustar = thistar_dict["tau_hat"]*self.ecmy.tau

        # else:
        #     cons = self.constraints_tau(ge_dict, id, ge=False, mil=True)
        #     bnds = self.bounds()
        #     thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, args=(self.b, np.array([id]), -1, False, ), method="SLSQP", options={"maxiter":mxit})

        return(thistar['x'])

    def ft_sv(self, id, ge_x):

        tau_hat_ft = 1 / self.ecmy.tau
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        tau_hat_sv = ge_dict["tau_hat"]
        tau_hat_sv[id, ] = tau_hat_ft[id, ]
        ge_dict_sv = self.ecmy.geq_solve(tau_hat_sv, np.ones(self.N))
        ge_x_sv = self.ecmy.unwrap_ge_dict(ge_dict_sv)

        return(ge_x_sv)


    def nft_sv(self, id, ge_x):

        tau_hat_nft = self.tau_nft / self.ecmy.tau
        np.fill_diagonal(tau_hat_nft, 1)
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        tau_hat_sv = ge_dict["tau_hat"]
        tau_hat_sv[id, ] = tau_hat_nft[id, ] # start slightly above free trade
        ge_dict_sv = self.ecmy.geq_solve(tau_hat_sv, np.ones(self.N))
        ge_x_sv = self.ecmy.unwrap_ge_dict(ge_dict_sv)

        return(ge_x_sv)

    def v_sv(self, id, ge_x, v):

        tau_v = np.tile(np.array([v]).transpose(), (1, self.N))
        tau_hat_v = (tau_v + .1) / self.ecmy.tau
        np.fill_diagonal(tau_hat_v, 1)
        ge_dict = self.ecmy.rewrap_ge_dict(copy.deepcopy(ge_x))
        tau_hat_sv = ge_dict["tau_hat"]
        tau_hat_sv[id, ] = tau_hat_v[id, ] # start slightly above free trade
        ge_dict_sv = self.ecmy.geq_solve(tau_hat_sv, np.ones(self.N))
        ge_x_sv = self.ecmy.unwrap_ge_dict(ge_dict_sv)

        return(ge_x_sv)

    def v_sv_all(self, v):

        tau_v = np.tile(np.array([v]).transpose(), (1, self.N))
        tau_hat_v = (tau_v + .1) / self.ecmy.tau
        np.fill_diagonal(tau_hat_v, 1)
        ge_dict_sv = self.ecmy.geq_solve(tau_hat_v, np.ones(self.N))
        ge_x_sv = self.ecmy.unwrap_ge_dict(ge_dict_sv)

        return(ge_x_sv)

    def export_results(self, out_dict, path):

        with open(path, 'w', newline="") as csv_file:
            writer = csv.writer(csv_file)
            for key, value in out_dict.items():
               writer.writerow([key, value])

    def import_results(self, path):

        with open(path) as csv_file:
            reader = csv.reader(csv_file)
            theta_dict = dict(reader)

        b = np.zeros(self.N)
        for key in theta_dict.keys():
            if key[0] == 'b':
                b[int(key[1])] = theta_dict[key]
            else:
                theta_dict[key] = float(theta_dict[key])
        keys_del = ['b' + str(i) for i in range(self.N)]
        for key in keys_del:
            try:
                del theta_dict[key]
            except KeyError:
                print("key not found")

        return(b, theta_dict)

    def br_cor(self, ge_x_sv, m, affinity, epsilon, b, theta_dict, mpec=True):
        """Best response correspondence. Given current policies, calculates best responses for all govs and returns new ge_x flattened vector.

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function for order of values.
        mpec : bool
            Calculate wrt mpec (true) or enforce equilibrium in calculation (false)

        Returns
        -------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs at best response values.

        """

        ge_dict_sv = self.ecmy.rewrap_ge_dict(ge_x_sv)
        tau_hat_sv = np.copy(ge_dict_sv["tau_hat"])
        ge_x = np.copy(ge_x_sv)
        tau_hat_nft = self.tau_nft / self.ecmy.tau
        np.fill_diagonal(tau_hat_nft, 1)

        tau_hat_br = ge_dict_sv["tau_hat"]
        wv = self.war_vals(b, m, theta_dict, epsilon) # calculate war values
        for id in range(self.N):
            wv_i = wv[:,id]
            ge_x_nft = self.nft_sv(id, ge_x)
            ge_x_i = self.br(ge_x_nft, b, wv_i, id, affinity=affinity)
            update = True
            tau_hat_i = self.ecmy.rewrap_ge_dict(ge_x_i)["tau_hat"][id, ]
            if np.any(np.isnan(ge_x_i)):
                update = False
                print("passing update...")
            if np.all(tau_hat_i==tau_hat_nft[id, ]):
                update = False
                print("passing update...")
            if update is True:
                tau_hat_br[id, ] = tau_hat_i

        ge_dict = self.ecmy.geq_solve(tau_hat_br, np.ones(self.N))
        ge_x = self.ecmy.unwrap_ge_dict(ge_dict)

        return(ge_x)

    def nash_eq(self, ge_x_sv, b, theta_dict, m, affinity):
        """Calculates Nash equilibrium of policy game

        Returns
        -------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs at NE values.

        """

        epsilon = np.zeros((self.N, self.N))

        # ge_x_sv = np.ones(self.ecmy.ge_x_len)
        ge_x_out = opt.fixed_point(self.br_cor, ge_x_sv, args=(m, affinity, epsilon, b, theta_dict, True, ), method="iteration", xtol=1e-02)
        return(ge_x_out)
