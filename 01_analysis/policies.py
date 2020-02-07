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
# import nlopt # NOTE: something wrong with build of this on laptop

# TODO: gradient to BR function

class policies:

    def __init__(self, data, params, ROWname, results_path=None, rcv_ft=False):
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
        self.M = data["M"]  # milex
        # self.rhoM = self.rho()  # loss of strength gradient

        self.tauMin = 1  # enforce lower bound on policies
        self.tauMax = 15
        self.tau_nft = 1.25  # where to begin search for best response

        self.hhat_len = self.N**2+4*self.N  # X, d, P, w, r, E
        self.Dhat_len = self.N
        self.tauj_len = self.N**2-self.N
        # self.lambda_i_len = self.hhat_len + self.tauj_len + 1 + self.N + (self.N - 1)  # ge vars, other policies, tau_ii, deficits, mil constraints
        # self.lambda_i_len = self.hhat_len + 1 + (self.N - 1)
        self.lambda_i_x_len = self.hhat_len + self.Dhat_len + self.tauj_len + 1 # one is own policy (redundant?)
        self.lambda_i_len = self.lambda_i_x_len + self.N
        # self.lambda_i_len_td = self.lambda_i_len + self.N ** 2 - self.N # add constraints on others' policies

        # NOTE: values less than zero seem to mess with best response
        # self.b_vals = np.arange(0, 1.1, .1)  # preference values for which to generate regime change value matrix.
        self.v_step = .1
        self.v_vals = np.arange(1, np.max(self.ecmy.tau), self.v_step)
        # self.v_vals = np.arange(1, 1.1, .1)
        self.v_max = np.max(self.ecmy.tau, axis=1) - .1
        np.savetxt(results_path + "v_vals.csv", self.v_vals, delimiter=",")

        self.x_len = self.ecmy.ge_x_len

        self.chi_min = .001
        # rcv_path = results_path + "rcv.csv"
        # if not os.path.isfile(rcv_path):
        #     rcv = self.pop_rc_vals(rcv_ft=rcv_ft)
        #     self.rc_vals_to_csv(rcv, rcv_path)
        #     self.rcv = rcv
        # else:
        #     self.rcv = self.read_rc_vals(rcv_path)

        ge_x_ft_path = results_path + "ge_x_ft.csv"
        if not os.path.isfile(ge_x_ft_path):
            self.ge_x_ft = np.zeros((self.N, self.x_len))
            for i in range(self.N):
                ge_x_ft_i = self.ft_sv(i, np.ones(self.x_len))
                ge_x_ft[i, ] = ge_x_ft_i
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
        # tau_mv[tau_mv < 0] = 0
        tau_mv = np.clip(tau_mv, 0, np.inf)
        r = np.sum(tau_mv * self.ecmy.Xcif, axis=1)

        return(r)

    def R_hat(self, ge_dict, v):

        v_mat = np.array([v])
        r = self.r_v(ge_dict, v)
        # print(r)
        if np.any(r == 0):
            print("r_v vector has zeros")
            print(r)

        tau_prime = ge_dict["tau_hat"] * self.ecmy.tau
        tau_prime_mv = tau_prime - np.tile(v_mat.transpose(), (1, self.N))
        tau_prime_mv = np.clip(tau_prime_mv, 0, np.inf)
        X_prime = ge_dict["X_hat"] * self.ecmy.Xcif
        r_prime = np.sum(tau_prime_mv * X_prime, axis=1)

        r_hat = r_prime / r
        # print(r_hat)
        # r_hat[r==0] = 0

        return(r_hat)

    def tau_diffs(self, tau_hat_x, tau_hat, id):
        """Short summary.

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs.
        tau_hat : matrix
            N times N array of initial tau_hats
        id : int
            government choosing policy

        Returns
        -------
        array
            Differences between proposed and initial tau_hat values for all govs not id

        """


        j = [x for x in range(self.N) if x != id]
        tau_hat_j = tau_hat[j, ]
        tau_hat_x_j = tau_hat_x[j, ]
        tau_diff_M = tau_hat_j - tau_hat_x_j

        out = []
        out.extend(tau_diff_M.ravel())

        return(np.array(out))

    def Lagrange_i_x(self, ge_x, v, tau_hat, war_vals, lambda_i_x, id):
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

        lambda_dict_i = self.rewrap_lambda_i(lambda_i_x)
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)

        G_hat_i = self.G_hat(ge_x, v, id, sign=1)
        geq_diffs = self.ecmy.geq_diffs(ge_x)
        tau_diffs = self.tau_diffs(ge_dict["tau_hat"], tau_hat, id)
        tau_ii_diff = ge_dict["tau_hat"][id, id] - 1
        D_diffs = ge_dict["D_hat"] - 1
        war_diffs = self.war_diffs(ge_x, v, war_vals, id)

        L_i = G_hat_i - np.dot(lambda_dict_i["h_hat"], geq_diffs) - np.dot(lambda_dict_i["D_hat"], D_diffs) - lambda_dict_i["tau_ii"] * tau_ii_diff - np.dot(lambda_dict_i["tau_hat"], tau_diffs) - np.dot(lambda_dict_i["chi_i"], war_diffs)

        return(L_i)

    def war_vals(self, v, m, theta_dict, c_bar=10):
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
        wv = np.clip(wv, 0, np.inf)

        return(wv)

    def rcv_ft(self, v):

        # TODO pre-compute ge_x and save so we don't have to iterate on geq_solve
        out = np.array([self.G_hat(self.ge_x_ft[i, ], v, 0, all=True) for i in range(self.N)])

        return(out.T)

    def Lzeros(self, ge_x_lbda_i_x, v, tau_hat, war_vals, id, enforce_geq=False, bound="lower"):
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

        x = ge_x_lbda_i_x[0:self.x_len]
        lambda_i_x = ge_x_lbda_i_x[self.x_len:]

        ge_dict = self.ecmy.rewrap_ge_dict(x)
        lambda_dict_i = self.rewrap_lambda_i(lambda_i_x)
        L_grad_f = ag.grad(self.Lagrange_i_x)
        L_grad = L_grad_f(x, v, tau_hat, war_vals, lambda_i_x, id)

        # L_grad_i = L_grad[self.N*id:self.N*(id+1)]
        # L_grad_h = L_grad[self.N**2+self.N:]  # skip policies and deficits
        # L_grad_out = []
        # L_grad_out.extend(L_grad_i)
        # L_grad_out.extend(L_grad_h)
        # L_grad_out = np.array(L_grad_out)
        # L_grad = L_grad_out

        tau_ii_diff = np.array([ge_dict["tau_hat"][id, id] - 1])

        if enforce_geq == True:
            tau_diffs = self.tau_diffs(ge_dict["tau_hat"], tau_hat, id)
            geq_diffs = self.ecmy.geq_diffs(x)
            D_diffs = ge_dict["D_hat"] - 1

        # calculate war constraints
        war_diffs = self.war_diffs(x, v, war_vals, id)

        out = []
        out.extend(L_grad)
        # if geq == True:
        #     out.extend(geq_diffs)
        if enforce_geq == True:
            out.extend(tau_diffs)
            out.extend(D_diffs)
            out.extend(geq_diffs)
        # if td == True:
        #     out.extend(tau_diffs)
        out.extend(tau_ii_diff)
        out.extend(war_diffs)  # NOTE: converted these to equality constraints, where negative values turned to zeros
        # out.extend(war_diffs * lambda_dict_i["chi_i"])  # complementary slackness

        if bound == "lower":
            return(np.array(out))
        else:
            return(np.array(out)*-1)

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
        theta_dict["c_hat"] = theta[self.N]
        theta_dict["alpha"] = theta[self.N+1]
        theta_dict["gamma"] = theta[self.N+2]

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
        x_lbda_theta_sv[self.x_len+self.lambda_i_len*self.N+self.N] = theta_dict_init["c_hat"]
        x_lbda_theta_sv[self.x_len+self.lambda_i_len*self.N+self.N+1] = theta_dict_init["alpha"]
        x_lbda_theta_sv[self.x_len+self.lambda_i_len*self.N+self.N+2] = theta_dict_init["gamma"]

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

    def Lzeros_loss(self, x_lbda_theta_sv):

        tau_hat = self.ecmy.rewrap_ge_dict(x_lbda_theta_sv[0:self.x_len])["tau_hat"]
        loss = 0
        for i in range(self.N):
            loss += self.loss_tau(tau_hat[i, ], i)

        return(loss)

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

        rhoM = self.rhoM(theta_dict, np.zeros((self.N, self.N)))
        m_diag = np.diagonal(m)
        m_frac = m / m_diag

        chi_logit = rhoM * m_frac ** theta_dict["gamma"]

        chi = chi_logit / (1 + chi_logit)

        return(chi)

    def rhoM(self, theta_dict, epsilon):
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
        rhoM = np.exp(-1 * (self.W * theta_dict["alpha"]) + epsilon)

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

    def D_diffs(self, ge_x):
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        return(ge_dict["D_hat"] - 1)

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

        if bound == "lower":
            return(x_L)
        else:
            return(x_U)

    def G_hat_wrap(self, v, id, sign):
        def f(x):
            return(self.G_hat(x, v, id, sign=sign))
        return(f)

    def br_ipyopt(self, v, id, wv=None):

        # verbose
        ipyopt.set_loglevel(ipyopt.LOGGING_DEBUG)

        x0 = self.v_sv(id, np.ones(self.x_len), v)
        # g_len = self.x_len - (self.N - 1)
        geq_c_len = self.x_len - self.N**2 - self.N
        if not wv is None:
            g_len = self.x_len - self.N**2
            g_upper = np.zeros(self.x_len - self.N**2)
            g_upper[geq_c_len:] = np.inf
            print(g_upper)
        else:
            g_len = geq_c_len

        g_sparsity_indices_a = np.array(np.meshgrid(range(g_len), range(self.x_len))).T.reshape(-1,2)
        g_sparsity_indices = (g_sparsity_indices_a[:,0], g_sparsity_indices_a[:,1])
        h_sparsity_indices_a = np.array(np.meshgrid(range(self.x_len), range(self.x_len))).T.reshape(-1,2)
        h_sparsity_indices = (h_sparsity_indices_a[:,0], h_sparsity_indices_a[:,1])

        problem = ipyopt.Problem(self.x_len, self.br_bounds_ipyopt(x0, id, "lower"), self.br_bounds_ipyopt(x0, id, "upper"), g_len, np.zeros(g_len), g_upper, g_sparsity_indices, h_sparsity_indices, self.G_hat_wrap(v, id, -1), self.G_hat_grad_ipyopt_wrap(v, id), self.br_cons_ipyopt_wrap(id, v, wv), self.br_cons_ipyopt_jac_wrap(id, v, wv))

        problem.set(print_level=5, nlp_scaling_method="none", fixed_variable_treatment='make_parameter')
        print("solving...")
        _x, obj, status = problem.solve(x0)

        return(_x, obj, status)

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
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        tau_hat_sv = ge_dict["tau_hat"]
        tau_hat_sv[id, ] = tau_hat_v[id, ] # start slightly above free trade
        ge_dict_sv = self.ecmy.geq_solve(tau_hat_sv, np.ones(self.N))
        ge_x_sv = self.ecmy.unwrap_ge_dict(ge_dict_sv)

        return(ge_x_sv)

    def loss_tau(self, tau_i, id, weights=None):
        """Loss function for b estimation, absolute log loss

        Parameters
        ----------
        tau_i : vector
            length N vector of i's policies
        weights : vector
            length N vector of weights on loss (most natural thing would be to weight based on gdp)

        Returns
        -------
        float
            loss

        """
        if weights is None:
            weights = np.ones(self.N)
        weights = weights / np.sum(weights)  # normalize

        tau_star = self.ecmy.tau[id, ] * tau_i

        out = np.sum((self.ecmy.tau[id, ] - tau_star) ** 2 * weights)

        return(out)

    def est_v_i_grid(self, id, v_init, m, theta_dict, epsilon):
        """Estimate b_i by searching over grid of possible values and calculating loss on predicted policy

        Parameters
        ----------
        id : int
            id of government for which to estimate preference parameter
        b_init : vector
            Length n vector of starting values for
        m : matrix
            N times N matrix of military deployments
        theta_dict : dict
            Dictionary storing values of military paramters
        epsilon : matrix
            N times N matrix of war shocks

        Returns
        -------
        float
            estimate for b_i

        """

        # endpoints for search
        # bmax = np.copy(np.max(self.b_vals))
        vmax = np.max(self.ecmy.tau[id, ]) - self.v_step
        # bmin = np.copy(np.min(self.b_vals))
        vmin = 1
        v_vec = np.copy(v_init)

        v = v_vec[id]

        # turn this on when we've reached local min
        stop = False

        # starting values (nearft)
        # ge_x_sv = self.nft_sv(id, np.ones(self.x_len))
        loss_last = 1000

        while stop is False:

            # turn these flags on if we're searching one of the lower or upper bounds of the b space
            lb = False
            ub = False

            # first
            idx_first = hp.which_nearest(self.v_vals, v)
            # start away from corners
            if idx_first == len(self.v_vals) - 1:
                idx_first -= 1
            if idx_first == 0:
                idx_first += 1
            idx_up = idx_first + 1
            idx_down = idx_first - 1

            # if either of the values to search is a bound, note this so we can check termination condition below
            if idx_up == len(self.v_vals) - 1:
                ub = True
            if idx_down == 0:
                lb = True

            v = self.v_vals[idx_first]

            Loss = []  # store values for local loss
            for idx in [idx_down, idx_first, idx_up]:
                v_idx = self.v_vals[idx]
                v_vec[id] = v_idx
                wv = self.war_vals(v_vec, m, theta_dict, epsilon) # calculate war values
                wv_i = wv[:,id]

                ge_x_sv = self.v_sv(id, np.ones(self.x_len), v_vec)

                br = self.br(ge_x_sv, v_vec, wv_i, id)  # calculate best response
                br_dict = self.ecmy.rewrap_ge_dict(br)
                tau_i = br_dict["tau_hat"][id, ]
                # loss = self.loss_tau(tau_i, id, weights=self.ecmy.Y)
                loss = self.loss_tau(tau_i, id)
                Loss.append(loss)

            # if median value is a valley, return it as estimate
            if np.argmin(Loss) == 1:
                stop = True
                # TODO: prevent binary search if local loss is less than previous loss
                # if loss_last < Loss[1]:
                #     b = b_max
                # else:
                #     stop = True
            else:  # otherwise, truncate search region and search in direction of of lower loss
                if np.argmin(Loss) == 2:
                    vmin = v
                    v = (vmax - v) / 2 + vmin
                if np.argmin(Loss) == 0:
                    vmax = v
                    v = (v - vmin) / 2 + vmin

            # check bounds, if loss decreases in direction of bound return bound as estimate
            if lb is True:
                if np.argmin(Loss) == 0:
                    v = self.v_vals[idx_down]
                    stop = True
            if ub is True:
                if np.argmin(Loss) == 2:
                    v = self.v_vals[idx_up]
                    stop = True

            # terminate if we've gotten stuck
            if np.abs(vmax - vmin) < .1:
                if np.argmin(Loss) == 0:
                    v = self.v_vals[idx_down]
                    stop = True
                elif np.argmin(Loss) == 2:
                    v = self.v_vals[idx_up]
                    stop = True
                else:
                    stop = True

        return(v)

    def est_v_grid(self, v_sv, m, theta_dict, epsilon, thres=.1):
        """Estimate vector of b values through grid search

        Parameters
        ----------
        b_sv : vector
            Length N vector of starting values for b
        m : matrix
            N times N matrix of military deployments
        theta_dict : dict
            Dictionary storing values of military paramters
        epsilon : matrix
            N times N matrix of war shocks
        thres : threshold for concluding search

        Returns
        -------
        vector
            Length N vector of preference paramter estimates

        """

        converged = False  # flag for convergence
        v_vec = np.copy(v_sv)

        while converged is False:
            v_out = np.copy(v_vec)  # current values of v
            for id in range(0, self.N):  # update values by iteratively calculating best estimates, holding other values fixed
                v_star = self.est_v_i_grid(id, v_vec, m, theta_dict, epsilon)
                v_vec[id] = v_star  # update vector
            if np.sum((v_vec - v_out) ** 2) < thres:  # if change in values less than threshold then terminate search
                converged = True

        return(v_vec)

    def epsilon_star(self, v, m, theta_dict):
        """Return critical epsilon (row's critical value for invading column, all epsilon greater than epsilon star will trigger invasion). If war costs exceed value of winning the war for sure then this value is infty

        Parameters
        ----------
        b : vector
            N times 1 vector of preference parameters
        m : matrix
            N times N matrix of military allocations
        theta_dict : dict
            military parameters
        W : matrix
            distance between belligerents

        Returns
        -------
        matrix
            N times N matrix of critical war shocks (zeros on diagonal)

        """

        m_diag = np.diagonal(m)
        m_frac = m / m_diag

        rcv = np.zeros((self.N, self.N))  # empty regime change value matrix (row's value for invading column)
        for i in range(self.N):
            v_nearest = hp.find_nearest(self.v_vals, v[i])
            rcv[i, ] = self.rcv[v_nearest][i, ]  # grab rcvs associated with b_nearest and extract ith row
            # (i's value for invading all others)

        # rcv = rcv.T

        out = theta_dict["alpha"] * self.W - theta_dict["gamma"] * np.log(m_frac) + np.log( 1 / ( theta_dict["c_hat"] ** -1 * (rcv - 1) - 1 ) )
        out[np.isnan(out)] = np.inf

        return(out)

    def weights(self, epsilon, sigma_epsilon):
        """Calculate weights for constraint regressions. If row, column constraint is active with some probability, then column's constraint vis a vis row will be included in regression.

        Parameters
        ----------
        epsilon : matrix
            N times N matrix of epsilon_star values (0s along diagonal)
        sigma_epsilon : float
            Variance of war shocks

        Returns
        -------
        matrix
            N times N matrix of weights (sum to 1)

        """

        out = 1 - stats.norm.cdf(epsilon, loc=0, scale=sigma_epsilon)
        out = out / np.sum(out)

        return(out)

    def G_lower(self, v, m, theta_dict):
        """Calculate j's utility when constraint vis a vis i is off

        Parameters
        ----------
        j : type
            Description of parameter `j`.
        i : type
            Description of parameter `i`.
        v : type
            Description of parameter `v`.
        m : type
            Description of parameter `m`.
        theta_dict : type
            Description of parameter `theta_dict`.

        Returns
        -------
        type
            Description of returned object.

        """

        wv = self.war_vals(v, m, theta_dict, np.zeros((self.N, self.N))) # calculate war values

        G_lower = np.zeros((self.N, self.N))
        for i in range(self.N):
            wv_i = wv[:,i]
            for j in range(self.N):
                wv_ij = copy.deepcopy(wv_i)
                wv_ij[j] = 0
                ge_x_sv = self.v_sv(i, np.ones(self.x_len), v)
                br_ij = self.br(ge_x_sv, v, wv_ij, i)
                G_ji = self.G_hat(br_ij, v, j)
                G_lower[j, i] = G_ji  # government i's welfare when constraint vis a vis j is lifted

        return(G_lower)

    def trunc_epsilon(self, epsilon_star, theta_dict):

        te = hp.mean_truncnorm(epsilon_star, theta_dict["sigma_epsilon"])
        te[np.isnan(te)] = 0

        return(te)

    def est_theta(self, X, Y):
        """Estimate military parameters from constraints. Iteratively recalculate parameters and weights until convergence.

        Parameters
        ----------
        fe : vector
            N times 1 vector of "fixed effects," expected payoffs in absence of coercion
        v : vector
            N times 1 vector of preference parameters
        m : matrix
            N times N matrix of military allocations
        theta_dict : dict
            Starting values of parameters (to calculate weights)

        Returns
        -------
        dict
            Updated parameter values

        """

        theta_out = np.zeros(2)
        ests = sm.OLS(Y, X, missing="drop").fit()
        theta_out[0] = ests.params[0]  # gamma
        theta_out[1] = -ests.params[1]  # alpha

        return(theta_out)

    def Y(self, rcv, theta_dict, G_ji):
        return(np.log( 1 / (theta_dict["c_hat"] ** -1 * (rcv - G_ji) - 1) ))

    def est_theta_inner(self, v, theta_dict, m, thres=.0001):

        m_diag = np.diagonal(m)
        m_frac = m / m_diag

        rcv = np.zeros((self.N, self.N))  # empty regime change value matrix (row's value for invading column)
        for i in range(self.N):
            v_nearest = hp.find_nearest(self.v_vals, v[i])
            rcv[i, ] = self.rcv[v_nearest][i, ]  # grab rcvs associated with b_nearest and extract ith row
            # (i's value for invading all others)

        diff = 10
        while diff > thres:

            theta_k0 = copy.deepcopy(np.array([theta_dict["gamma"], theta_dict["alpha"]]))
            epsilon_star = self.epsilon_star(v, m, theta_dict)
            G_lower = self.G_lower(v, m, theta_dict)
            Y_lower = self.Y(rcv, theta_dict, G_lower)
            t_epsilon = self.trunc_epsilon(epsilon_star, theta_dict)

            lhs = self.Y(rcv, theta_dict, 1)
            phi = stats.norm.cdf(epsilon_star.ravel(), loc=0, scale=theta_dict["sigma_epsilon"])
            Y = lhs.ravel() - (1 - phi.ravel()) * t_epsilon.ravel() - phi.ravel() * Y_lower.ravel()
            X = np.column_stack( (1 - phi.ravel()) * (np.log(m_frac.ravel()), (1 - phi.ravel()) * self.W.ravel()))

            X_active = X[~np.isnan(X[:,0]),:]
            Y_active = Y[~np.isnan(X[:,0])]

            theta_k1 = self.est_theta(X_active, Y_active)
            theta_dict["gamma"] = theta_k1[0]
            theta_dict["alpha"] = theta_k1[1]

            diff = np.sum(np.abs(theta_k1 - theta_k0))

        return(theta_dict)

    def est_theta_outer(self, v, theta_dict_init):

        return(0)

    def est_loop_interior(self, v_init, theta_dict_init, thres=.001):
        """For fixed values of epsilon and c_hat, estimate preference parameters and alpha, gamma

        Parameters
        ----------
        b_init : vector
            N times 1 vector of initial preference parameters
        theta_dict_init : dict
            Dictionary storing values of alpha, gamma, c_hat, sigma_epsilon
        thres : float
            Convergence criterion for inner loop

        """

        m = self.M / np.ones((self.N, self.N))
        m = m.T
        m[self.ROW_id,:] = 0
        m[:,self.ROW_id] = 0
        m[self.ROW_id,self.ROW_id] = 1

        v_k = np.copy(v_init)
        theta_dict_k = copy.deepcopy(theta_dict_init)

        diffs = 10
        k = 1
        while diffs > thres:

            v_km1 = np.copy(v_k)
            theta_km1 = np.copy(np.array([i for i in theta_dict_k.values()]))
            vals_km1 = np.append(v_km1, theta_km1)

            epsilon = np.zeros((self.N, self.N))
            v_k = self.est_v_grid(v_k, m, theta_dict_k, epsilon)
            theta_dict_k = self.est_theta_inner(v_k, theta_dict_k, m)

            theta_k = np.array([i for i in theta_dict_k.values()])
            vals_k = np.append(v_k, theta_k)

            diffs = np.sum((vals_k - vals_km1) ** 2)
            k += 1

        Loss_k = 0
        for id in range(self.N):

            # war values
            wv = self.war_vals(v_k, m, theta_dict_k, np.zeros((self.N, self.N))) # calculate war values
            wv_i = wv[:,id]

            # starting values
            ge_x_sv = self.nft_sv(id, np.ones(self.x_len))

            br = self.br(ge_x_sv, v_k, wv_i, id)  # calculate best response
            br_dict = self.ecmy.rewrap_ge_dict(br)
            tau_i = br_dict["tau_hat"][id, ]
            # Loss_k += self.loss_tau(tau_i, id, weights=self.ecmy.Y)
            Loss_k += self.loss_tau(tau_i, id)

        return(v_k, theta_dict_k, Loss_k)

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

    def unwrap_xy_tau(self, dict_xy_tau):
        """choice variables, multipliers, and preference parameters dictionary to flattened vector

        Parameters
        ----------
        dict_xy_tau : dict
            dictionary storing choice variables, multipliers, and preference parameters

        Returns
        -------
        vector
            flattened vector of choice variables, multipliers, and preference parameters

        """

        x = []
        x.extend(dict_xy_tau["ge_x"])  # ge vars (x, h)
        x.extend(dict_xy_tau["lambda_x"])
        x.extend(dict_xy_tau["b"])

        return(np.array(x))

    def rewrap_xy_tau(self, xy_tau):
        """choice variables, multipliers, and preference parameters flattened vector to dictionary

        Parameters
        ----------
        xy_tau : vector
            flattened vector of choice variables, multipliers, and preference parameters

        Returns
        -------
        dict
            dictionary storing choice variables, multipliers, and preference parameters

        """

        dict_xy_tau = dict()
        dict_xy_tau["ge_x"] = xy_tau[0:self.x_len]
        dict_xy_tau["lambda_x"] = xy_tau[self.x_len:self.x_len+self.lambda_i_len*self.N]
        dict_xy_tau["b"] = xy_tau[self.x_len+self.lambda_i_len*self.N:self.x_len+self.lambda_i_len*self.N+self.N]

        return(dict_xy_tau)

    def unwrap_x(self, dict_x):
        """Take dictionary of sub-vectors of br_m input and return flattened vector

        Parameters
        ----------
        dict_x : dict
            Dictionary storing sub-components of br_m input

        Returns
        -------
        vector
            Flattened array of self.br_m inputs

        """

        x = []
        x.extend(dict_x["ge_x"])
        x.extend(dict_x["lambda_x"])
        x.extend(dict_x["m_x"])

        return(np.array(x))

    def rewrap_x(self, x):
        """Take br_m input vector and return dictionary of three sub-vectors:
            1. ge_x: ge vars and policies (use economy.rewrap_ge_dict to get dictionary)
            2. lambda_x: multipliers (use policies.rewrap_lambda to get dictionary)
            3. m_x: military allocations (use policies.rewrap_m to get matrix)

        Parameters
        ----------
        x : vector
            Flattened array of self.br_m inputs

        Returns
        -------
        dict
            Dictionary storing sub-components of x

        """

        ge_len = self.ecmy.ge_x_len
        lambda_len = self.N*self.lambda_i_len
        m_len = self.N**2

        dict_x = dict()
        dict_x["ge_x"] = x[0:ge_len]
        dict_x["lambda_x"] = x[ge_len:ge_len+lambda_len]
        dict_x["m_x"] = x[ge_len+lambda_len:ge_len+lambda_len+m_len]

        return(dict_x)

    def unwrap_y(self, dict_y):
        """Convert dictionary of estimation targets to flattened vector

        Parameters
        ----------
        dict_y : dict
            Dictionary storing estimation targets

        Returns
        -------
        vector
            Flattened vector of estimation targets

        """

        y = []
        y.extend(dict_y["theta_m"])
        y.extend(dict_y["m"])
        y.extend(dict_y["lambda_M"])
        y.extend(dict_y["lambda_x"])

        return(np.array(y))

    def rewrap_y(self, y):
        """Convert flattened vector of estimation targets to dictionary

        Parameters
        ----------
        y : vector
            Flattened vector of estimation targets

        Returns
        -------
        dict
            Dictionary storing estimation targets

        """

        dict_y = dict()
        dict_y["theta_m"] = y[0:self.theta_len]
        dict_y["m"] = y[self.theta_len:self.theta_len+self.N**2]
        dict_y["lambda_M"] = y[self.theta_len+self.N**2:self.theta_len+self.N**2+self.N]
        dict_y["lambda_x"] = y[self.theta_len+self.N**2+self.N:self.theta_len+self.N**2+self.N+self.lambda_i_len*self.N]

        return(dict_y)

    def unwrap_theta(self, dict_theta):
        """Convert dictionary storing military structural paramters to flattened vector

        Parameters
        ----------
        dict_theta : dict
            Dictionary of military structural parameters

        Returns
        -------
        vector
            Flattened vector of military structural parameters

        """

        theta = []
        theta.extend(dict_theta["b"])
        theta.extend(dict_theta["alpha"])
        # theta.extend(dict_theta["gamma"])
        theta.extend(dict_theta["c_hat"])

        return(np.array(theta))

    def rewrap_theta(self, theta):
        """Convert flattened vector of military structural parameters to dictionary

        Parameters
        ----------
        theta : vector
            Flattened vector of military structural parameters

        Returns
        -------
        dict
            Dictionary of military structural parameters

        """

        dict_theta = dict()
        dict_theta["b"] = theta[0:self.N]
        dict_theta["alpha"] = theta[self.N:self.N+self.alpha_len]
        # dict_theta["gamma"] = theta[self.N+self.alpha_len:self.N+self.alpha_len+1]
        # dict_theta["c_hat"] = theta[self.N+self.alpha_len+1:self.N+self.alpha_len+2]
        dict_theta["c_hat"] = theta[self.N+self.alpha_len:self.N+self.alpha_len+1]

        return(dict_theta)

    def unwrap_m(self, m):
        """Convert military allocation matrix into flattened array

        Parameters
        ----------
        m : matrix
            N times N matrix of military deployments.

        Returns
        -------
        vector
            Flattened array of military deployments

        """

        x = []
        for i in range(self.N):
            x.extend(m[i, ])
        return(np.array(x))

    def rewrap_m(self, m_x):
        """Convert miliary allocation vector into matrix

        Parameters
        ----------
        m_x : vector
            Flattened array of military deployments

        Returns
        -------
        type
            N times N matrix of military deployments.

        """
        m = np.reshape(m_x, (self.N, self.N))
        return(m)

    def unwrap_lambda_i(self, lambda_dict_i):
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
        x.extend(lambda_dict_i["D_hat"])
        x.extend(lambda_dict_i["tau_hat"])
        x.extend(lambda_dict_i["tau_ii"])
        x.extend(lambda_dict_i["chi_i"])

        return(np.array(x))

    def rewrap_lambda_i(self, x):
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
        lambda_dict_i["D_hat"] = x[self.hhat_len:self.hhat_len+self.Dhat_len]  # ge vars
        lambda_dict_i["tau_hat"] = x[self.hhat_len+self.Dhat_len:self.hhat_len+self.Dhat_len+self.tauj_len]
        lambda_dict_i["tau_ii"] = x[self.hhat_len+self.Dhat_len+self.tauj_len:self.hhat_len+self.Dhat_len+self.tauj_len+1]  # own policy contraint
        lambda_dict_i["chi_i"] = x[self.hhat_len+self.Dhat_len+self.tauj_len+1:]  # mil constraints, threats against i

        return(lambda_dict_i)
