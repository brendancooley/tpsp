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
import sys

class policies:

    def __init__(self, data, params, ROWname):
        """

        Parameters
        ----------
        data : dict
            dictionary storing data inputs
        params : dict
            dictionary storing economic primitive parameters
        ROWname : str
            name of rest of world aggregate ("RoW")

        """

        # Setup economy
        self.ecmy = economy.economy(data, params)
        self.N = self.ecmy.N
        self.ids = np.arange(self.N)

        self.ROW_id = np.where(data["ccodes"]==ROWname)[0][0]
        self.ccodes = data["ccodes"]

        # purge deficits
        self.ecmy.purgeD()  # replicate DEK

        # enforce positive tariffs
        tau_hat_pos = np.ones_like(self.ecmy.tau)
        tau_hat_pos[self.ecmy.tau < 1] = 1 / self.ecmy.tau[self.ecmy.tau < 1]
        self.ecmy.update_ecmy(tau_hat_pos, np.ones(self.N), np.ones(self.N))

        self.W = data["W"]  # distances
        self.W = self.W / np.min(self.W[self.W>1])  # normalize
        np.fill_diagonal(self.W, 1)
        # np.fill_diagonal(self.W, 0)  # EU has positive own distance by averaging over members, zero this out
        self.M = data["M"]  # milex

        # construct equal distribution m matrix, enforce zeros on ROW row and columns
        self.m = self.M / np.ones((self.N, self.N))
        self.m = self.m.T
        # self.m[self.ROW_id,:] = 0
        # self.m[:,self.ROW_id] = 0
        # self.m[self.ROW_id,self.ROW_id] = 1

        self.Y_norm = self.ecmy.Y / np.min(self.ecmy.Y)
        # NOTE: may want to do this with W as well

        # counterfactual m, no threats
        self.mzeros = np.diag(self.M)

        self.tau_nft = 1.25  # near free trade policies, starting values to begin search for best response

        # len of different variable subsets
        self.hhat_len = self.N**2+4*self.N  # X, d, P, w, r, E
        self.Dhat_len = self.N
        self.tauj_len = self.N**2-self.N  # non i policies
        # self.lambda_i_len = self.hhat_len + self.N  # ge multipliers plus mil multipliers
        self.lambda_i_len = self.hhat_len

        # optimizer parameters
        self.x_len = self.ecmy.ge_x_len  # len of ge vars
        # self.xlshvt_len = self.x_len + self.lambda_i_len * self.N + self.N**2 + self.N + 4  # ge vars, all lambdas (flattened), slack variables, v, theta (except v)
        # self.xlshvt_len = self.x_len + self.lambda_i_len * self.N + self.N**2 + self.hhat_len*self.N + self.N + 4
        self.xlhvt_len = self.x_len + self.lambda_i_len * self.N + self.hhat_len*self.N + self.N + 5 + self.N
        # self.xlsh_len = self.x_len + self.lambda_i_len + self.N + self.hhat_len
        self.xlh_len = self.x_len + self.lambda_i_len + self.hhat_len

        # self.g_len = self.hhat_len + (self.hhat_len + self.N - 1)*self.N + self.N**2 + self.N**2  #     constraints len: ge_diffs, Lzeros (own policies N-1), war_diffs mat, comp_slack mat
        # self.g_len = self.hhat_len + (self.hhat_len + self.N - 1)*self.N + self.N**2 + self.N**2 + self.hhat_len*self.N
        self.g_len = self.hhat_len + (self.hhat_len + self.N - 1)*self.N + self.hhat_len*self.N
        # self.L_i_len = self.x_len + self.lambda_i_len + self.N
        # self.L_i_len = self.x_len + self.lambda_i_len + self.N + self.hhat_len
        self.L_i_len = self.x_len + self.lambda_i_len + self.hhat_len

        self.max_iter_ipopt = 100000
        self.chi_min = 1.0e-4  # minimum value for chi
        self.wv_min = -1.0e2  # minimum war value
        self.alpha1_ub = self.alpha1_min(.01)  # restrict alpha search (returns alpha such that rho(alpha)=.01)
        self.zero_lb_relax = -1.0e-30  # relaxation on zero lower bound for ipopt (which are enforced without slack by ipopt (see 0.15 NLP in ipopt options))
        self.mu_min = 1.0e-200
        self.v_min = .5
        self.tau_buffer = .5

        self.tick = 0  # tracker for optimization calls to loss function

    def G_hat(self, x, v, id, sign=1, all=False):
        """Calculate changes in government welfare given ge inputs and outputs

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function ecmy.unwrap_ge_dict for order of values.
        v : vector
            len N array of preference values for each government
        id : int
            id of government to return welfare welfare changes for
        sign : scalar
            Scales output. Use to convert max problems to min.
        all : bool
            return all govs' welfare changes if this flag is True

        Returns
        -------
        vector or float
            G changes for selected governments

        """

        ge_dict = self.ecmy.rewrap_ge_dict(x)
        # Uhat = self.ecmy.U_hat(ge_dict, v)
        Uhat = self.ecmy.U_hat(ge_dict, v)
        # print(Uhat)
        # Ghat = Uhat * self.R_hat(ge_dict, v)
        # Ghat = .75 * np.log(Uhat) + .25 * np.log(R_hat)
        Ghat = Uhat

        if all == False:
            return(Ghat[id]*sign)
        else:
            return(Ghat*sign)

    def G_hat_grad(self, x, v, id, sign):
        """Calculate gradient of G_hat (autograd wrapper)

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function ecmy.unwrap_ge_dict for order of values.
        v : vector
            len N array of preference values for each government
        id : int
            id of government to return welfare welfare changes for
        sign : scalar
            Scales output. Use to convert max problems to min.

        Returns
        -------
        vector
            vector of len self.x_len of partial derivatives of objective function

        """
        G_hat_grad_f = ag.grad(self.G_hat)
        return(G_hat_grad_f(x, v, id, sign))

    def G_hat_tilde(self, ge_x, h, id, m, v, theta_dict):
        """Calculate war-risk inclusive utility change

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function ecmy.unwrap_ge_dict for order of values.
        h : vector
            vector length self.hhat of regime change ge response vars
        id : int
            Gov id for which to calculate probability of survival
        m : matrix
            self.N times self.N matrix of military allocations
        v : vector
            len self.N array of preference values for each government
        theta_dict : dict
            dictionary of military parameters (see self.rewrap_theta)

        Returns
        -------
        float
            Utility change at g_x times probability of survival

        """

        H_i = self.H(ge_x, h, id, m, v, theta_dict)
        G_i = self.G_hat(ge_x, v, id)
        # C_i = self.C(ge_x, h, id, m, v, theta_dict)
        # print(C_i)
        rcx = self.rcx(self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"], h, id)
        G_i_ft = self.G_hat(rcx, v, id)

        # out = G_i - C_i
        if id != self.ROW_id:
            out = H_i * G_i + (1 - H_i) * G_i_ft
        else:
            out = G_i
        # out = np.log(H_i) + np.log(G_i)
        # out = np.log(H_i) + G_i

        return(out)

    def r_v(self, v):
        """Calculate factual government revenue for given values of v

        Parameters
        ----------
        v : vector
            len N array of preference values for each government

        Returns
        -------
        vector
            len N array of v-adjusted government revenues evaluated at self.ecmy.tau

        """

        v_mat = np.array([v])
        tau_mv = self.ecmy.tau - np.tile(v_mat.transpose(), (1, self.N))
        tau_mv = tau_mv - np.diag(np.diag(tau_mv))
        r = np.sum(tau_mv * self.ecmy.Xcif, axis=1)

        return(r)

    def R_hat(self, ge_dict, v):
        """Calculate change in government revenue given ge values and v

        Parameters
        ----------
        ge_dict : dict
            Dictionary storing ge inputs and outputs
        v : vector
            len N array of preference values for each government

        Returns
        -------
        vector
            len N array of v-adjusted changes in government revenues

        """

        v_mat = np.array([v])
        r = self.r_v(v)

        tau_prime = ge_dict["tau_hat"] * self.ecmy.tau
        tau_prime_mv = tau_prime - np.tile(v_mat.transpose(), (1, self.N))
        tau_prime_mv = tau_prime_mv - np.diag(np.diag(tau_prime_mv))
        X_prime = ge_dict["X_hat"] * self.ecmy.Xcif
        r_prime = np.sum(tau_prime_mv * X_prime, axis=1)

        r_hat = r_prime / r
        # r_hat = np.clip(r_hat, 0, np.inf)

        return(r_hat)

    def Lagrange_i_x(self, ge_x, h, lambda_i_x, id, m, v, theta_dict):
        """Short summary.

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function ecmy.unwrap_ge_dict for order of values.
        h : vector
            vector length self.hhat of regime change ge response vars
        lambda_i_x : vector
            ge and for gov id
        id : int
            government choosing policy
        m : matrix
            self.N times self.N matrix of military allocations
        v : vector
            len N array of preference values for each government
        theta_dict : dict (see self.rewrap_theta)
            dictionary of military parameters

        Returns
        -------
        float
            Lagrangian evaluated at ge_x, lambda_i_x

        """

        # lambda_dict_i = self.rewrap_lbda_i(lambda_i_x)
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        rcx = self.rcx(ge_dict["tau_hat"], h, id)

        # G_hat_i = self.G_hat(ge_x, v, id, sign=1)
        G_hat_tilde_i = self.G_hat_tilde(ge_x, h, id, m, v, theta_dict)

        geq_diffs = self.ecmy.geq_diffs(ge_x, v)
        # h_diffs = self.ecmy.geq_diffs(rcx)

        # wv = self.wv_rcx(rcx, id, m, v, theta_dict)
        # war_diffs = self.war_diffs(ge_x, v, wv, id)

        # wd = -1 * war_diffs  # flip these so violations are positive

        # L_i = G_hat_i - np.dot(lambda_dict_i["h_hat"], geq_diffs) - np.dot(lambda_dict_i["chi_i"], wd)
        L_i = G_hat_tilde_i - np.dot(lambda_i_x, geq_diffs)

        return(L_i)

    def Lzeros_i(self, xlh, id, m, v, theta_dict):
        """calculate first order condition for goverment i, ge_x_lbda_i input

        Parameters
        ----------
        ge_x_lbda_i : vector
            len self.x_len + self.lambda_i_len vector storing concatenated ge_x and lbda_i values
        id : int
            id of government choosing policy vector
        m : matrix
            self.N times self.N matrix of military allocations
        v : vector
            len N array of preference values for each government
        theta_dict : dict (see self.rewrap_theta)
            dictionary of military parameters


        Returns
        -------
        vec
            len self.N (policies) + self.hhat_len (ge_vars) vector storing partial derivatives of government id's Lagrangian

        """

        xlh_dict = self.rewrap_xlh(xlh)

        ge_x = xlh_dict["ge_x"]
        lambda_i_x = xlh_dict["lambda_i"]
        # s_i = xlsh_dict["s_i"]
        h = xlh_dict["h"]

        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        L_grad_f = ag.grad(self.Lagrange_i_x)
        # ge_x_rch = np.concatenate((ge_x, h))
        L_grad = L_grad_f(ge_x, h, lambda_i_x, id, m, v, theta_dict)
        inds = self.L_grad_i_ind(id)
        L_grad_i = L_grad[inds]

        out = []
        out.extend(L_grad_i)

        return(np.array(out))

    def L_grad_i_ind(self, id):
        """which elements of full Lagrange gradient to return for government id. (drops tau_{-i} and deficits)

        Parameters
        ----------
        id : int
            government for which to calculate indicators

        Returns
        -------
        vector
            vector length self.x_len of indices for gov id's first order conditions

        """

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

    def war_diffs(self, ge_x, v, war_vals, id):
        """calculate difference between government id's utility at proposed vector ge_x versus war value

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function ecmy.unwrap_ge_dict for order of values.
        v : vector
            len self.N array of preference values for each government
        war_vals : vector
            Length self.N vector of war values for each non-id country in war against id
        id : int
            Gov id for which to calculate war constraints

        Returns
        -------
        vector
            Length self.N vector of peace minus war against id utility differences

        """

        G = self.G_hat(ge_x, v, 0, all=True)
        war_diffs = G - war_vals

        return(war_diffs)

    def chi(self, m, theta_dict):
        """calculate war success probability matrix

        Parameters
        ----------
        m : matrix
            self.N times self.N matrix of military allocations
        theta_dict : dict
            dictionary of military parameters (see self.rewrap_theta)

        Returns
        -------
        matrix
            self.N times self.N matrix of war success probabilities

        """

        rho = self.rho(theta_dict)  # loss of strength matrix
        m_diag = np.diagonal(m)
        m_frac = m / m_diag
        # m_frac = np.zeros((self.N, self.N))

        chi_logit = rho * m_frac ** theta_dict["gamma"]  # logit transformation
        chi = chi_logit / (1 + chi_logit)  # transform back

        return(chi)

    def rho(self, theta_dict):
        """calculate loss of strength gradient given theta, data (self.W)

        Parameters
        ----------
        theta_dict : dict
            Dictionary storing military structural parameters

        Returns
        -------
        matrix
            N times N matrix loss of strength gradient

        """

        rho = np.exp(-1 * (theta_dict["alpha0"] + self.W * theta_dict["alpha1"]))
        rho -= np.diag(np.diag(rho))  # set diagonal to one

        return(rho)

    def C(self, ge_x, h, id, m, v, theta_dict):

        # Cinv = theta_dict["C"] ** -1
        # Cinv_i = np.array([Cinv[i] for i in range(self.N) if i != id])
        # Cinv_i = theta_dict["c_hat"] ** -1
        # Cinv = np.delete(Cinv, id)
        eta = theta_dict["eta"]
        gamma = theta_dict["gamma"]
        alpha = theta_dict["alpha1"]

        m_diag = np.diagonal(m)
        m_frac = m / m_diag
        np.fill_diagonal(m_frac, 0)

        m_frac_i = np.array([m_frac[:,id][i] for i in range(self.N) if i != id])
        W_i = np.array([self.W[:,id][i] for i in range(self.N) if i != id])

        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        rcx = self.rcx(ge_dict["tau_hat"], h, id)
        G = self.G_hat(ge_x, v, id, all=True)
        rcv = self.G_hat(rcx, v, id, all=True)
        DeltaG = rcv - G
        # print(DeltaG)
        DeltaG = np.clip(DeltaG, 1e-10, np.inf)
        DeltaG = np.array([DeltaG[i] for i in range(self.N) if i != id])
        print(id)
        print(DeltaG)

        out = np.sum(theta_dict["c_hat"] * m_frac_i**gamma * W_i**(-alpha) * DeltaG**eta)

        return(out)

    def H(self, ge_x, h, id, m, v, theta_dict):
        """Compute probability nobody attacks government id

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function ecmy.unwrap_ge_dict for order of values.
        h : vector
            vector length self.hhat of regime change ge response vars
        id : int
            Gov id for which to calculate probability of survival
        m : matrix
            self.N times self.N matrix of military allocations
        v : vector
            len self.N array of preference values for each government
        theta_dict : dict
            dictionary of military parameters (see self.rewrap_theta)

        Returns
        -------
        float
            probability government id is not attacked

        """

        # pr_peace_ji = np.exp(chi_ji)
        chi_ji = self.peace_probs(ge_x, h, id, m, v, theta_dict)[0]
        pr_peace = np.exp(np.sum(chi_ji))
        # print("----")
        # print(pr_peace)

        return(pr_peace)

    def peace_probs(self, ge_x, h, id, m, v, theta_dict):

        # for i in theta_dict.keys():
        #     print(i)
        #     print(theta_dict[i])

        # Cinv = theta_dict["C"] ** -1
        # Cinv_i = np.array([Cinv[i] for i in range(self.N) if i not in [id, self.ROW_id]])

        # C_mat = np.outer(1 / theta_dict["C"], theta_dict["C"])
        # C_mat_i = np.array([C_mat[id,:][i] for i in range(self.N) if i != id])
        Cinv_i = theta_dict["c_hat"] ** -1
        # Cinv = np.delete(Cinv, id)
        eta = theta_dict["eta"]
        gamma = theta_dict["gamma"]
        alpha = theta_dict["alpha1"]
        alpha2 = theta_dict["alpha2"]

        m_diag = np.diagonal(m)
        m_frac = m / m_diag
        np.fill_diagonal(m_frac, 0)

        m_frac_i = np.array([m_frac[:,id][i] for i in range(self.N) if i not in [id, self.ROW_id]])
        # print(m_frac_i)
        W_i = np.array([self.W[:,id][i] for i in range(self.N) if i not in [id, self.ROW_id]])
        Y_i = np.array([self.Y_norm[i] for i in range(self.N) if i not in [id, self.ROW_id]])

        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        rcx = self.rcx(ge_dict["tau_hat"], h, id)
        G = self.G_hat(ge_x, v, id, all=True)
        rcv = self.G_hat(rcx, v, id, all=True)
        DeltaG = rcv - G
        # print(DeltaG)
        DeltaG = np.clip(DeltaG, 1e-10, np.inf)
        DeltaG = np.array([DeltaG[i] for i in range(self.N) if i not in [id, self.ROW_id]])
        # DeltaG[id] = 0
        # print(Cinv)
        # print(m_frac[:,id]**gamma)
        # print(self.W[:,id]**(-1*alpha))
        # print(DeltaG**eta)

        # chi_ji = -Cinv_i * m_frac_i**gamma * W_i**(-1*alpha) * DeltaG**eta
        chi_ji = -Cinv_i * Y_i**alpha2 * m_frac_i**gamma * W_i**(-1*alpha) * DeltaG**eta
        # chi_ji = - m_frac[:,id]**gamma * self.W[:,id]**(-1*alpha) * DeltaG**eta
        # print(chi_ji)

        return(chi_ji, np.exp(chi_ji))

    def rewrap_xlhvt(self, xlhvt):
        """Convert flattened xlhvt vector to dictionary

        Parameters
        ----------
        xlhvt : vector (see self.unwrap_xlhvt)
            len self.xlhvt_len vector storing ge vars, lambdas (flattened), vs, theta

        Returns
        -------
        dict
            dictionary of flattened vectors for each sub element of xlhvt

        """

        xlhvt_dict = dict()
        xlhvt_dict["ge_x"] = xlhvt[0:self.x_len]
        xlhvt_dict["lbda"] = xlhvt[self.x_len:self.x_len+self.lambda_i_len*self.N]  # np.reshape(xlshvt_dict["lbda"] (self.N, self.lambda_i_len)) gives matrix of lambdas, 1 row for each government
        # xlshvt_dict["s"] = xlshvt[self.x_len+self.lambda_i_len*self.N:self.x_len+self.lambda_i_len*self.N+self.N**2]
        xlhvt_dict["h"] = xlhvt[self.x_len+self.lambda_i_len*self.N:self.x_len+self.lambda_i_len*self.N+self.N*self.hhat_len]
        xlhvt_dict["v"] = xlhvt[self.x_len+self.lambda_i_len*self.N+self.N*self.hhat_len:self.x_len+self.lambda_i_len*self.N+self.N*self.hhat_len+self.N]
        xlhvt_dict["theta"] = xlhvt[self.x_len+self.lambda_i_len*self.N+self.N*self.hhat_len+self.N:]

        return(xlhvt_dict)

    def unwrap_xlhvt(self, xlhvt_dict):
        """Convert xlhvt dictionary into flattened vector

        Parameters
        ----------
        xlhvt_dict : dict (see self.rewrap_xlhvt)
            dictionary of flattened vectors for each sub element of xlhvt

        Returns
        -------
        vector
            len self.xlhvt_len vector storing ge vars, lambdas (flattened), vs, theta

        """

        xlhvt = []
        xlhvt.extend(xlhvt_dict["ge_x"])
        xlhvt.extend(xlhvt_dict["lbda"])
        # xlshvt.extend(xlshvt_dict["s"])
        xlhvt.extend(xlhvt_dict["h"])
        xlhvt.extend(xlhvt_dict["v"])
        xlhvt.extend(xlhvt_dict["theta"])

        return(np.array(xlhvt))

    def rewrap_theta(self, theta_x):
        """Convert theta dictionary into flattened vector

        Parameters
        ----------
        theta_x : vector (see self.unwrap_theta)
            flattened vector of military parameters

        Returns
        -------
        dict
            dictionary of military parameters

        """

        theta_dict = dict()
        # theta_dict["c_hat"] = theta_x[0]
        theta_dict["eta"] = theta_x[0]
        theta_dict["gamma"] = theta_x[1]
        # theta_dict["alpha0"] = theta_x[2]  # baseline power projection loss
        theta_dict["c_hat"] = theta_x[2]
        theta_dict["alpha1"] = theta_x[3]  # distance coefficient
        theta_dict["alpha2"] = theta_x[4]  # gdp coefficient
        theta_dict["C"] = theta_x[5:5+self.N]

        return(theta_dict)

    def unwrap_theta(self, theta_dict):
        """Convert theta vector into dictionary

        Parameters
        ----------
        theta_dict : dict (see self.rewrap_theta)
            dictionary of military parameters

        Returns
        -------
        vector
            flattened vector of military parameters

        """

        theta_x = []
        # theta_x.extend(np.array([theta_dict["c_hat"]]))
        theta_x.extend(np.array([theta_dict["eta"]]))
        theta_x.extend(np.array([theta_dict["gamma"]]))
        # theta_x.extend(np.array([theta_dict["alpha0"]]))
        theta_x.extend(np.array([theta_dict["c_hat"]]))
        theta_x.extend(np.array([theta_dict["alpha1"]]))
        theta_x.extend(np.array([theta_dict["alpha2"]]))
        theta_x.extend(np.array(theta_dict["C"]))

        return(np.array(theta_x))

    def unwrap_lbda_i(self, lambda_dict_i):
        """Convert dictionary of multipliers for gov i into vector for Lagrangians

        Parameters
        ----------
        lambda_dict_i : dict (see self.rewrap_lbda_i)
            dictionary storing multipliers for government i's ge constraints and mil constraints

        Returns
        -------
        vector
            vector length self.h_hat_len + self.N (ge constraints and i's mil constraints)

        """

        x = []
        x.extend(lambda_dict_i["h_hat"])
        x.extend(lambda_dict_i["chi_i"])

        return(np.array(x))

    def rewrap_lbda_i(self, x):
        """Return dictionary of Lagrange multipliers from vector of multipliers for given gov id

        Parameters
        ----------
        x : vector
            vector length self.h_hat_len + self.N (ge constraints and i's mil constraints)

        Returns
        -------
        dict
            dictionary storing multipliers for government i's ge constraints and mil constraints

        """

        lambda_dict_i = dict()
        lambda_dict_i["h_hat"] = x[0:self.hhat_len]  # ge vars
        lambda_dict_i["chi_i"] = x[self.hhat_len:]  # mil constraints, threats against i

        return(lambda_dict_i)

    def loss(self, xlhvt):
        """Calculate policy loss.

        Parameters
        ----------
        xlhvt : vector (see self.unwrap_xlhvt)
            len self.xlhvt_len vector storing ge vars, lambdas (flattened), vs, theta

        Returns
        -------
        float
            loss

        """

        # alpha = .1

        xlhvt_dict = self.rewrap_xlhvt(xlhvt)

        ge_x = xlhvt_dict["ge_x"]
        v = xlhvt_dict["v"]
        theta_x = xlhvt_dict["theta"]
        # print(theta_x)

        # optimizer tracker
        self.tick += 1
        if self.tick % 50 == 0:  # print output every 25 calls

            # s = np.reshape(xlshvt_dict["s"], (self.N, self.N))
            # print("s:")
            # print(s)
            h = np.reshape(xlhvt_dict["h"], (self.N, self.hhat_len))
            ge_dict = self.ecmy.rewrap_ge_dict(ge_x)

            cons = self.estimator_cons(xlhvt, self.m)

            print("max constraint violation:")
            print(np.argmax(np.abs(cons)))
            print(np.max(np.abs(cons)))

            print("tau:")
            print(ge_dict["tau_hat"]*self.ecmy.tau)
            print("-----")

            print("ge dict:")
            print(ge_dict)
            print("-----")

            print("v:")
            print(v)

            theta_dict = self.rewrap_theta(theta_x)
            print("theta_dict:")
            for i in theta_dict.keys():
                print(i)
                print(theta_dict[i])
            # print(self.rewrap_theta(theta_x))

            print("G_hat:")
            print(self.G_hat(ge_x, v, 0, all=True))

            print("R_hat:")
            print(self.R_hat(self.ecmy.rewrap_ge_dict(ge_x), v))
            print("r_v:")
            print(self.r_v(v))

            for i in range(self.N):
                # print("lambda chi " + str(i))
                lbda = np.reshape(self.rewrap_xlhvt(xlhvt)["lbda"], (self.N, self.lambda_i_len))
                # lbda_chi_i = self.rewrap_lbda_i(lbda[i, ])["chi_i"]
                # print(lbda_chi_i)
                # print("s " + str(i))
                # print(s[i, ])
                h_i = h[i, ]
                print("peace probs " + str(i) + ":")
                peace_probs_i = self.peace_probs(ge_x, h_i, i, self.m, v, self.rewrap_theta(theta_x))[1]
                print(peace_probs_i)

            sys.stdout.flush()

        tau_hat = self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"]
        tau_star = tau_hat * self.ecmy.tau  # calculate equilibrium tau

        tau_diffs = tau_star - self.ecmy.tau  # difference with factual tau
        loss = np.sum(tau_diffs**2)  # euclidean norm
        # loss = np.sum(np.abs(tau_diffs))  # 1-norm
        # loss = np.sum(np.where(np.abs(tau_diffs) < alpha, tau_diffs**2, np.abs(tau_diffs))) # smooth l-1 loss

        return(loss)

    def loss_grad(self, xlhvt, out):
        """gradient of loss function (autograd wrapper)

        Parameters
        ----------
        xlhvt : vector (see self.unwrap_xlhvt)
            len self.xlhvt_len vector storing ge vars, lambdas (flattened), vs, theta
        out : vector
            len self.xlhvt_len vector to store output (requirement for ipopt)

        Returns
        -------
        vector
            vector of partial derivatives of loss with respect to xlhvt inputs

        """

        loss_grad_f = ag.grad(self.loss)
        out[()] = loss_grad_f(xlhvt)

        return(out)

    def geq_diffs_xlhvt(self, xlhvt):
        """calculate differences between ge inputs and tau_hat-consistent ge output (ge constraints satisfied when this returns zero vector)

        Parameters
        ----------
        xlhvt : vector (see self.unwrap_xlhvt)
            len self.xlhvt_len vector storing ge vars, lambdas (flattened), vs, theta

        Returns
        -------
        vector
            len self.h_hat_len vector of differences between ge vars in xlhvt and tau_hat-consistent ge_output

        """
        ge_x = self.rewrap_xlhvt(xlhvt)["ge_x"]
        v = self.rewrap_xlhvt(xlhvt)["v"]
        geq_diffs = self.ecmy.geq_diffs(ge_x, v)
        return(geq_diffs)

    def h_diffs_xlhvt(self, xlhvt, id):
        """calculate differences regime change vectors and current trial h values

        Parameters
        ----------
        xlhvt : vector (see self.unwrap_xlhvt)
            len self.xlhvt_len vector storing ge vars, lambdas (flattened), vs, theta

        Returns
        -------
        vector
            len self.h_hat_len vector of differences between trial h values and rcx-consistent ge_output

        """
        xlhvt_dict = self.rewrap_xlhvt(xlhvt)
        ge_x = xlhvt_dict["ge_x"]
        v = xlhvt_dict["v"]
        tau_hat_tilde = self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"]
        h_i = np.reshape(xlhvt_dict["h"], (self.N, self.hhat_len))[id, ]
        rcx = self.rcx(tau_hat_tilde, h_i, id)
        h_diffs = self.ecmy.geq_diffs(rcx, v)
        return(h_diffs)

    def Lzeros_i_xlhvt(self, xlhvt, id, m):
        """calculate first order condition for government i, xlhvt input (wrapper around self.L_zeros_i for estimation)

        Parameters
        ----------
        xlhvt : vector (see self.unwrap_xlhvt)
            len self.xlhvt_len vector storing ge vars, lambdas (flattened), vs, theta
        id : int
            id for country for which to calculate FOC
        m : matrix
            self.N times self.N matrix of military allocations

        Returns
        -------
        vector
            len self.N (policies) + self.hhat_len (ge_vars) vector storing partial derivatives of government id's Lagrangian

        """

        xlhvt_dict = self.rewrap_xlhvt(xlhvt)
        ge_x = xlhvt_dict["ge_x"]
        lbda = np.reshape(xlhvt_dict["lbda"], (self.N, self.lambda_i_len))
        lbda_i = lbda[id, ]
        # s = np.reshape(xlshvt_dict["s"], (self.N, self.N))
        # s_i = s[id, ]
        v = xlhvt_dict["v"]
        theta_x = xlhvt_dict["theta"]
        theta_dict = self.rewrap_theta(theta_x)
        # print(theta_dict)

        tau_hat_tilde = self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"]
        h_i = np.reshape(xlhvt_dict["h"], (self.N, self.hhat_len))[id, ]
        # rcx = self.rcx(tau_hat_tilde, h_i, id)

        xlh = np.concatenate((ge_x, lbda_i, h_i))

        # wv = self.wv_rcx(rcx, id, m, v, theta_dict)  # calculate war values

        Lzeros_i = self.Lzeros_i(xlh, id, m, v, theta_dict)  # flattened and concatenated ge vars and lbda_i as input

        return(Lzeros_i)

    # def war_diffs_xlshvt(self, xlshvt, id, m):
    #     """calculate values of war constraints for government id, given ge vars and parameters in xlshvt (wrapper around self.war_diffs)
    #
    #     Parameters
    #     ----------
    #     xlshvt : vector (see self.unwrap_xlshvt)
    #         len self.xlshvt_len vector storing ge vars, lambdas (flattened), vs, theta
    #     id : id
    #         id of government for which to calculate war diffs
    #     m : matrix
    #         self.N times self.N matrix of military allocations
    #
    #     Returns
    #     -------
    #     vector
    #         length self.N vector of war diffs
    #
    #     """
    #
    #     xlshvt_dict = self.rewrap_xlshvt(xlshvt)
    #     ge_x = xlshvt_dict["ge_x"]
    #     s = np.reshape(xlshvt_dict["s"], (self.N, self.N))
    #     s_i = s[id, ]
    #     v = xlshvt_dict["v"]
    #     theta_x = xlshvt_dict["theta"]
    #     theta_dict = self.rewrap_theta(theta_x)
    #     # print(theta_dict)
    #
    #     tau_hat_tilde = self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"]
    #     h_i = np.reshape(xlshvt_dict["h"], (self.N, self.hhat_len))[id, ]
    #     rcx = self.rcx(tau_hat_tilde, h_i, id)
    #
    #     wv = self.wv_rcx(rcx, id, m, v, theta_dict)
    #     # wv = np.clip(wv, 0, np.inf)
    #
    #     war_diffs_i = self.war_diffs(ge_x, v, wv, id) - s_i
    #
    #     return(war_diffs_i)
    #
    # def comp_slack_xlshvt(self, xlshvt, id, m):
    #     """calculate value of complementary slackness conditions
    #
    #     Parameters
    #     ----------
    #     xlshvt : vector (see self.unwrap_xlshvt)
    #         len self.xlshvt_len vector storing ge vars, lambdas (flattened), vs, theta
    #     id : id
    #         id of government for which to calculate comp slack condition
    #     m : matrix
    #         self.N times self.N matrix of military allocations
    #
    #     Returns
    #     -------
    #     vector
    #         length self.N vector of complementary slackness conditions for government id
    #
    #     """
    #
    #     xlshvt_dict = self.rewrap_xlshvt(xlshvt)
    #
    #     ge_x = xlshvt_dict["ge_x"]
    #     lbda = np.reshape(xlshvt_dict["lbda"], (self.N, self.lambda_i_len))
    #     lbda_i = lbda[id, ]
    #
    #     s = np.reshape(xlshvt_dict["s"], (self.N, self.N))
    #     s_i = s[id, ]
    #
    #     # v = xlshvt_dict["v"]
    #     # theta_x = xlshvt_dict["theta"]
    #     # theta_dict = self.rewrap_theta(theta_x)
    #
    #     # wv = self.war_vals(v, m, theta_dict)
    #     # war_diffs_i = self.war_diffs(ge_x, v, wv[:,id], id)
    #
    #     lbda_i_chi = self.rewrap_lbda_i(lbda_i)["chi_i"]
    #     # comp_slack_i = war_diffs_i * lbda_i_chi
    #     comp_slack_i = s_i * lbda_i_chi
    #
    #     return(comp_slack_i)

    def estimator_cons(self, xlhvt, m):
        """return flattened vector of estimation constraints (length self.g_len). Equality and inequality included. Equality constraints set to zero. Inequality constraints greater than zero. These values set in self.estimator.

        Parameters
        ----------
        xlhvt : vector (see self.unwrap_xlhvt)
            len self.xlhvt_len vector storing ge vars, lambdas (flattened), vs, theta
        m : matrix
            self.N times self.N matrix of military allocations

        Returns
        -------
        vector
            length self.g_len of estimation constraints

        """

        # print(m)
        # geq constraints
        geq_diffs = self.geq_diffs_xlhvt(xlhvt)

        # Lagrange gradient
        Lzeros = []
        # war_diffs = []
        # comp_slack = []
        h_diffs = []
        for i in range(self.N):
            Lzeros_i = self.Lzeros_i_xlhvt(xlhvt, i, m)
            Lzeros.extend(Lzeros_i)
            # war_diffs_i = self.war_diffs_xlshvt(xlshvt, i, m)
            # war_diffs.extend(war_diffs_i)
            # comp_slack_i = self.comp_slack_xlshvt(xlshvt, i, m)
            # comp_slack.extend(comp_slack_i)
            h_diffs_i = self.h_diffs_xlhvt(xlhvt, i)
            h_diffs.extend(h_diffs_i)
        Lzeros = np.array(Lzeros)
        # war_diffs = np.array(war_diffs)
        # comp_slack = np.array(comp_slack)
        h_diffs = np.array(h_diffs)

        # out = np.concatenate((geq_diffs, Lzeros, war_diffs, comp_slack, h_diffs))
        out = np.concatenate((geq_diffs, Lzeros, h_diffs))

        return(out)

    def estimator_cons_wrap(self, m):
        """wrapper around self.estimator_cons for ipopt. Returns function that maps input and out vector into new values for constraints.

        Parameters
        ----------
        m : matrix
            self.N times self.N matrix of military allocations

        Returns
        -------
        funtion
            function suitable for evaluating constraints in ipopt.

        """

        def f(x, out):
            out[()] = self.estimator_cons(x, m)
            return(out)

        return(f)

    def estimator_cons_jac(self, xlhvt, m):
        """calculate constraint Jacobian (autograd wrapper)

        Parameters
        ----------
        xlhvt : vector (see self.unwrap_xlhvt)
            len self.xlhvt_len vector storing ge vars, lambdas (flattened), vs, theta
        m : matrix
            self.N times self.N matrix of military allocations

        Returns
        -------
        vector
            flattened self.xlhvt_len times self.g_len matrix of constraint jacobian values

        """

        # print(m)
        # print(xlhvt)
        geq_diffs_jac_f = ag.jacobian(self.geq_diffs_xlhvt)
        geq_diffs_jac = geq_diffs_jac_f(xlhvt)

        Lzeros_i_jac_f = ag.jacobian(self.Lzeros_i_xlhvt)
        # war_diffs_i_jac_f = ag.jacobian(self.war_diffs_xlshvt)
        # comp_slack_i_jac_f = ag.jacobian(self.comp_slack_xlshvt)
        h_diffs_i_jac_f = ag.jacobian(self.h_diffs_xlhvt)

        Lzeros_jac_flat = []
        # war_diffs_jac_flat = []
        # comp_slack_flat = []
        h_diffs_jac_flat = []
        for i in range(self.N):
            Lzeros_i_jac = Lzeros_i_jac_f(xlhvt, i, m)
            Lzeros_jac_flat.extend(Lzeros_i_jac.ravel())
            # war_diffs_i_jac = war_diffs_i_jac_f(xlshvt, i, m)
            # war_diffs_jac_flat.extend(war_diffs_i_jac.ravel())
            # comp_slack_i_jac = comp_slack_i_jac_f(xlshvt, i, m)
            # comp_slack_flat.extend(comp_slack_i_jac.ravel())
            h_diffs_i_jac = h_diffs_i_jac_f(xlhvt, i)
            h_diffs_jac_flat.extend(h_diffs_i_jac.ravel())

        # out = np.concatenate((geq_diffs_jac.ravel(), Lzeros_jac_flat, war_diffs_jac_flat, comp_slack_flat, h_diffs_jac_flat), axis=None)
        out = np.concatenate((geq_diffs_jac.ravel(), Lzeros_jac_flat, h_diffs_jac_flat), axis=None)

        # jac_f = ag.jacobian(self.estimator_cons)
        # out = jac_f(xlshvt, m).ravel()

        return(out)

    def estimator_cons_jac_wrap(self, m):
        """wrapper around self.estimator_cons_jac for ipopt. Returns function suitable for evaluating constraint Jacobian within ipopt.

        Parameters
        ----------
        g_sparsity_bin : vector
            flattened boolean array of indices to include in return (default: all true)
        m : matrix
            self.N times self.N matrix of military allocations

        Returns
        -------
        function
            function suitable for evaluating constraint Jacobian in ipopt

        """

        def f(x, out):
            out[()] = self.estimator_cons_jac(x, m)
            return(out)

        return(f)

    def geq_lb(self):

        tau_min_mat = copy.deepcopy(self.ecmy.tau)
        np.fill_diagonal(tau_min_mat, 5)

        lb_dict = dict()
        lb_dict["tau_hat"] = np.reshape(np.repeat(np.min(tau_min_mat - self.tau_buffer, axis=1), self.N), (self.N, self.N)) / self.ecmy.tau
        # lb_dict["tau_hat"] = np.reshape(np.repeat(0, self.N**2), (self.N, self.N))
        # lb_dict["tau_hat"] = self.v_min / self.ecmy.tau
        # lb_dict["tau_hat"] = 1. / self.ecmy.tau
        np.fill_diagonal(lb_dict["tau_hat"], 1)
        lb_dict["D_hat"] = np.repeat(1, self.N)
        lb_dict["X_hat"] = np.reshape(np.repeat(0, self.N**2), (self.N, self.N))
        lb_dict["P_hat"] = np.repeat(0, self.N)
        lb_dict["w_hat"] = np.repeat(0, self.N)
        lb_dict["r_hat"] = np.repeat(-np.inf, self.N)
        # lb_dict["r_hat"] = np.repeat(0, self.N)
        lb_dict["E_hat"] = np.repeat(0, self.N)

        out = self.ecmy.unwrap_ge_dict(lb_dict)

        return(out)

    def geq_ub(self):

        ub_dict = dict()
        ub_dict["tau_hat"] = np.reshape(np.repeat(np.inf, self.N**2), (self.N, self.N))
        ub_dict["tau_hat"] = np.reshape(np.repeat(np.max(self.ecmy.tau + self.tau_buffer, axis=1), self.N), (self.N, self.N)) / self.ecmy.tau
        # ub_dict["tau_hat"] = np.reshape(np.repeat(np.max(self.ecmy.tau, axis=1), self.N), (self.N, self.N)) / self.ecmy.tau
        np.fill_diagonal(ub_dict["tau_hat"], 1)
        ub_dict["D_hat"] = np.repeat(1, self.N)
        ub_dict["X_hat"] = np.reshape(np.repeat(np.inf, self.N**2), (self.N, self.N))
        ub_dict["P_hat"] = np.repeat(np.inf, self.N)
        ub_dict["w_hat"] = np.repeat(np.inf, self.N)
        ub_dict["r_hat"] = np.repeat(np.inf, self.N)
        ub_dict["E_hat"] = np.repeat(np.inf, self.N)

        out = self.ecmy.unwrap_ge_dict(ub_dict)

        return(out)

    def theta_bounds(self, bound="lower"):

        c_lb = 10
        c_ub = np.inf
        # alpha_lb = -1.
        # alpha_ub = 2.

        theta_dict_lb = dict()
        theta_dict_lb["eta"] = 1
        theta_dict_lb["gamma"] = -.5
        theta_dict_lb["c_hat"] = c_lb
        theta_dict_lb["alpha1"] = -1.0  # distance coefficient
        theta_dict_lb["alpha2"] = -.5  # gdp coefficient
        theta_dict_lb["C"] = np.repeat(c_lb, self.N)
        lb = self.unwrap_theta(theta_dict_lb)

        theta_dict_ub = dict()
        theta_dict_ub["eta"] = 1
        theta_dict_ub["gamma"] = 2.
        theta_dict_ub["c_hat"] = c_ub
        theta_dict_ub["alpha1"] = 1. # distance coefficient
        theta_dict_ub["alpha2"] = 1.75  # gdp coefficient
        theta_dict_ub["C"] = np.repeat(c_ub, self.N)
        ub = self.unwrap_theta(theta_dict_ub)

        if bound=="lower":
            return(lb)
        else:
            return(ub)

    def estimator_bounds(self, theta_x, v, bound="lower", nash_eq=False):
        """return bounds on input variables for estimator

        Parameters
        ----------
        bound : str
            "lower" or "upper", return upper or lower bounds on input values
        nash_eq : bool
            if true, constrain all parameters to initial values
        theta_x : vector
            initial values for theta
        v : vector
            initial values for v

        Returns
        -------
        vector
            length self.xlhvt_len vector of lower or upper bounds for input values

        """

        theta_dict = self.rewrap_theta(theta_x)

        # set initial values for bounds
        x_L = np.repeat(-np.inf, self.xlhvt_len)
        x_U = np.repeat(np.inf, self.xlhvt_len)

        # bound tau_hats below at zero
        b = self.x_len
        x_L[0:b] = self.geq_lb()
        x_U[0:b] = self.geq_ub()

        # lbda_i_bound_dict = dict()
        # lbda_i_bound_dict["h_hat"] = np.repeat(-np.inf, self.hhat_len)
        # lbda_i_bound_dict["chi_i"] = np.repeat(self.zero_lb_relax, self.N)

        # lbda_i_bound_dict["chi_i"] = np.zeros(self.N)  # constrain inequality constraint multipliers to be positive
        # lbda_i_bound = self.unwrap_lbda_i(lbda_i_bound_dict)
        # lbda_bound = np.tile(lbda_i_bound, self.N)
        # print(lbda_bound)

        # constraints and slack variables
        # x_L[b:b+self.lambda_i_len*self.N] = lbda_bound  # mil constraint multipliers
        # x_L[b:b+self.lambda_i_len*self.N] = 0
        b += self.lambda_i_len*self.N
        # x_L[b:b+self.N**2] = 0  # positive slack variables
        # b += self.N**2

        # rcx
        rch_L = np.tile(self.geq_lb()[-self.hhat_len:], self.N)
        rch_U = np.tile(self.geq_ub()[-self.hhat_len:], self.N)

        x_L[b:b+self.N*self.hhat_len] = rch_L
        x_U[b:b+self.N*self.hhat_len] = rch_U
        b += self.N*self.hhat_len

        if nash_eq == False:  # set lower bounds on parameters, of fix some values for testing estimator
            x_L[b:b+self.N] = self.v_min # vs
            b += self.N
            theta_lb = self.theta_bounds("lower")
            theta_ub = self.theta_bounds("upper")
            theta_len = len(theta_lb)
            x_L[b:b+theta_len] = theta_lb
            x_U[b:b+theta_len] = theta_ub
        else:  # fix all parameters at initial values
            x_L[b:b+self.N] = v
            x_U[b:b+self.N] = v
            b += self.N
            # x_L[b] = theta_dict["c_hat"]  # c_hat
            # x_U[b] = theta_dict["c_hat"]
            x_L[b] = theta_dict["eta"]  # eta
            x_U[b] = theta_dict["eta"]
            b += 1
            x_L[b] = theta_dict["gamma"]  # gamma
            x_U[b] = theta_dict["gamma"]
            b += 1
            # x_L[b] = theta_dict["alpha0"]
            # x_U[b] = theta_dict["alpha0"]
            x_L[b] = theta_dict["c_hat"]
            x_U[b] = theta_dict["c_hat"]
            b += 1
            x_L[b] = theta_dict["alpha1"]
            x_U[b] = theta_dict["alpha1"]
            b += 1
            x_L[b:b+self.N] = theta_dict["C"]
            x_U[b:b+self.N] = theta_dict["C"]

        if bound == "lower":
            return(x_L)
        else:
            return(x_U)

    def v_max(self):

        sv = np.ones(self.N)
        out = opt.root(self.r_v, sv)

        return(out['x'])

    def alpha1_min(self, thres):
        """calculate minimum value for alpha1 (distance elasticity)

        Parameters
        ----------
        thres : float
            minimum desired rho, evaluated at minimum nonzero distance in data

        Returns
        -------
        float
            minimum value for alpha1

        """

        Wmin = np.min(self.W[self.W>0])
        alpha1_min = - np.log(thres) / Wmin

        return(alpha1_min)

    def pp_wrap_C(self, C, thres):

        Cinv = C ** -1
        chi_ji = -Cinv
        pr_peace = np.exp(chi_ji)

        out = pr_peace - thres

        return(out)

    def pp_wrap_alpha(self, alpha, thres):

        chi_ji = -np.min(self.W[self.W>1])**(-1*alpha)
        pr_peace = np.exp(chi_ji)

        out = pr_peace - thres

        return(out)

    def estimator_sv(self, m, v, theta_x, nash_eq=False):

        theta_dict = self.rewrap_theta(theta_x)

        # if nash_eq == True:
        #     ge_x_sv = self.v_sv_all(v)
        # else:
        #     ge_x_sv = np.ones(self.x_len)
        ge_x_sv = self.v_sv_all(v)
        # ge_x_sv = np.ones(self.x_len)

        lambda_sv = np.zeros(self.lambda_i_len*self.N)
        # lambda_sv = np.ones(self.lambda_i_len*self.N)
        # lambda_sv = np.repeat(.01, self.lambda_i_len*self.N)

        h_sv = []
        # s_sv = []
        for i in range(self.N):

            ft_id = self.ft_sv(i, ge_x_sv, v)
            # ft_id = self.ecmy.rewrap_ge_dict(self.ft_sv(i, ge_x_sv))
            # h_sv_i = self.ecmy.unwrap_ge_dict(self.ecmy.geq_solve(ft_id["tau_hat"], np.ones(self.N)))[-self.hhat_len:]
            # print(ft_id)
            if type(ft_id) is not int:
                h_sv_i = ft_id[-self.hhat_len:]
            else:
                print("cannot compute regime change ge vars for " + self.ccodes[i])
                h_sv_i = np.ones(self.hhat_len)
            h_sv.extend(h_sv_i)

            # rcx = self.rcx(np.ones((self.N, self.N)), h_sv_i, i)
            # wv_i = self.wv_rcx(rcx, i, m, v, theta_dict)
            # wd = self.war_diffs(ge_x_sv, v, wv_i, i)
            # s_sv.extend(wd)

        # h_sv = np.array(h_sv)
        # s_sv = np.array(s_sv)
        s_sv = np.zeros(self.N**2)

        v_sv = v
        theta_sv = theta_x

        # out = np.concatenate((ge_x_sv, lambda_sv, s_sv, h_sv, v_sv, theta_sv))
        out = np.concatenate((ge_x_sv, lambda_sv, h_sv, v_sv, theta_sv))

        return(out)

    def estimator(self, v_sv, theta_x_sv, m, sv=None, nash_eq=False):
        """estimate the model

        Parameters
        ----------
        v_sv : vector
            length self.N vector of staring values for preference parameters
        theta_x_sv : vector
            flattened vector of starting values for military parameters
        m : matrix
            self.N times self.N matrix of military allocations
        nash_eq : bool
            compute nash_equilibrium, fixing parameters, in lieu of estimating model

        Returns
        -------
        x, obj, status
            ipopt standard returns

        """

        x_len = self.xlhvt_len

        wd_g = np.repeat(np.inf, self.N**2)
        g_lower = np.zeros(self.g_len)
        g_upper = np.zeros(self.g_len)
        # g_upper[self.hhat_len + (self.hhat_len + self.N - 1)*self.N:self.hhat_len + (self.hhat_len + self.N - 1)*self.N+self.N**2] = wd_g

        # xlshvt_sv_dc = np.concatenate((np.ones(self.x_len), np.repeat(.01, self.lambda_i_len*self.N), np.zeros(self.N**2), v_sv, theta_x_sv))  # NOTE: for derivative checker, we will use these to calculate Jacobian sparsity
        # xlshvt_sv = np.concatenate((np.ones(self.x_len), np.zeros(self.lambda_i_len*self.N), np.zeros(self.N**2), v_sv, theta_x_sv)) # initialize starting values
        if sv is None:
            xlhvt_sv = self.estimator_sv(m, v_sv, theta_x_sv, nash_eq=nash_eq) # initialize starting values
        else:
            xlhvt_sv = sv

        # Jacobian sparsity (none)
        g_sparsity_indices_a = np.array(np.meshgrid(range(self.g_len), range(x_len))).T.reshape(-1,2)
        g_sparsity_indices = (g_sparsity_indices_a[:,0], g_sparsity_indices_a[:,1])
        g_sparsity_bin = np.repeat(True, self.g_len*self.xlhvt_len)

        # Lagrangian Hessian sparsity (none)
        h_sparsity_indices_a = np.array(np.meshgrid(range(self.xlhvt_len), range(self.xlhvt_len))).T.reshape(-1,2)
        h_sparsity_indices = (h_sparsity_indices_a[:,0], h_sparsity_indices_a[:,1])

        if nash_eq == False:
            b_L = self.estimator_bounds(theta_x_sv, v_sv, bound="lower")
            b_U = self.estimator_bounds(theta_x_sv, v_sv, bound="upper")
        else:
            b_L = self.estimator_bounds(theta_x_sv, v_sv, bound="lower", nash_eq=True)
            b_U = self.estimator_bounds(theta_x_sv, v_sv, bound="upper", nash_eq=True)

        if nash_eq == False:
            problem = ipyopt.Problem(self.xlhvt_len, b_L, b_U, self.g_len, g_lower, g_upper, g_sparsity_indices, h_sparsity_indices, self.loss, self.loss_grad, self.estimator_cons_wrap(m), self.estimator_cons_jac_wrap(m))
            # problem.set(print_level=5, fixed_variable_treatment='make_parameter', max_iter=self.max_iter_ipopt, constr_viol_tol=1.0e-05, dual_inf_tol=1.0, tol=1.0e-05, replace_bounds="yes")
            problem.set(print_level=5, fixed_variable_treatment='make_parameter', max_iter=self.max_iter_ipopt, nlp_scaling_method="none", mu_strategy="adaptive", mu_oracle="probing", fixed_mu_oracle="probing", adaptive_mu_restore_previous_iterate="yes", constr_viol_tol=1.0e-05, dual_inf_tol=1.0, tol=1.0e-05, mu_min=self.mu_min)
            # problem.set(print_level=5, fixed_variable_treatment='make_parameter', max_iter=self.max_iter_ipopt, linear_solver="pardiso")
            # for derivative test, make sure we don't travel too far from initial point with point_perturbation_radius (leads to evaluation errors)
            # problem.set(print_level=5, fixed_variable_treatment='make_parameter', max_iter=self.max_iter_ipopt, derivative_test="first-order", point_perturbation_radius=0.)
        else:
            problem = ipyopt.Problem(self.xlhvt_len, b_L, b_U, self.g_len, g_lower, g_upper, g_sparsity_indices, h_sparsity_indices, self.dummy, self.dummy_grad, self.estimator_cons_wrap(m), self.estimator_cons_jac_wrap(m))
            problem.set(print_level=5, fixed_variable_treatment='make_parameter', max_iter=self.max_iter_ipopt, linear_solver="pardiso", mu_strategy="adaptive", mu_oracle="probing", fixed_mu_oracle="probing", adaptive_mu_restore_previous_iterate="yes")
            # problem.set(print_level=5, fixed_variable_treatment='make_parameter', max_iter=self.max_iter_ipopt, linear_solver="pardiso", derivative_test="first-order", point_perturbation_radius=0.)
        print("solving...")
        _x, obj, status = problem.solve(xlhvt_sv)

        return(_x, obj, status)

    def rewrap_xlh(self, xlh):
        """convert Lagrange zeros vector to dictionary

        Parameters
        ----------
        xlh : vector
            vector length self.x_len + self.lambda_i_len + self.N + self.hhat of ge vars, i's multipliers, and regime change h vars

        Returns
        -------
        dict
            dictionary of ge vars, i's multipliers, i's slack variables, and regime change h vars

        """

        out = dict()
        out["ge_x"] = xlh[0:self.x_len]
        out["lambda_i"] = xlh[self.x_len:self.x_len+self.lambda_i_len]
        # out["s_i"] = xlsh[self.x_len+self.lambda_i_len:self.x_len+self.lambda_i_len+self.N]
        out["h"] = xlh[self.x_len+self.lambda_i_len:self.x_len+self.lambda_i_len+self.hhat_len]

        return(out)

    def unwrap_xlh(self, xlh_dict):
        """conver Lagrange zeros dictionary to vector

        Parameters
        ----------
        xlh_dict : dict
            dictionary of ge vars, i's multipliers, and regime change h vars

        Returns
        -------
        vector
            vector length self.x_len + self.lambda_i_len + self.hhat_len of ge vars, i's multipliers, and regime change h vars

        """

        x = []
        x.extend(xlh_dict["ge_x"])
        x.extend(xlh_dict["lambda_i"])
        # x.extend(xlh_dict["s_i"])
        x.extend(xlh_dict["h"])

        return(np.array(x))

    def wv_rcx(self, rcx, id, m, v, theta_dict):
        """calculate war values for Lagrange problem

        Parameters
        ----------
        rcx : vector
            length self.x_len vector of regime change ge vars
        id : int
            id of government for which to calculate best response
        m : matrix
            self.N times self.N matrix of military allocations
        v : vector
            vector length self.N of governments preference parameters
        theta_dict : dict (see self.rewrap_theta)
            dictionary of military parameters

        Returns
        -------
        type
            Description of returned object.

        """

        cv = self.G_hat(rcx, v, id, all=True)  # conquest value for all against id

        chi = self.chi(m, theta_dict)
        chi = np.clip(chi, self.chi_min, 1)
        # print(theta_dict["c_hat"])
        wc = theta_dict["c_hat"] / chi

        wv = cv - wc[:,id]  # cost for each country for invading id
        wv = np.clip(wv, 0, np.inf)
        # print(wv)

        return(wv)

    def Lzeros_i_cons(self, xlh, id, m, v, theta_dict):
        """Constraints for best response (Lagrangian version)

        # Parameters
        ----------
        xlh : vector
            vector length self.xlh_len of ge vars, id's multipliers, and regimge change vars
        id : int
            id of government for which to calculate best response
        m : matrix
            self.N times self.N matrix of military allocations
        v : vector
            vector length self.N of governments preference parameters
        theta_dict : dict (see self.rewrap_theta)
            dictionary of military parameters

        Returns
        -------
        vector
            vector length self.hhat_len + self.lambda_i_len + self.hhat_len of geq_diffs, Lagrange zeros, and rc_ge diffs

        """

        xlh_dict = self.rewrap_xlh(xlh)

        ge_x = xlh_dict["ge_x"]
        lambda_i_x = xlh_dict["lambda_i"]
        # s_i = xlsh_dict["s_i"]
        h = xlh_dict["h"]

        tau_hat_tilde = self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"]
        rcx = self.rcx(tau_hat_tilde, h, id)
        # wv = self.wv_rcx(rcx, id, m, v, theta_dict)

        geq_diffs = self.ecmy.geq_diffs(ge_x, v)
        # war_diffs = self.war_diffs(ge_x, v, wv, id)
        Lzeros = self.Lzeros_i(xlh, id, m, v, theta_dict)

        # comp_slack = s_i * self.rewrap_lbda_i(lambda_i_x)["chi_i"]
        h_diffs = self.ecmy.geq_diffs(rcx, v)

        # wd = war_diffs - s_i

        # out = np.concatenate((geq_diffs, Lzeros, wd, comp_slack, h_diffs))
        out = np.concatenate((geq_diffs, Lzeros, h_diffs))

        return(out)

    def Lzeros_i_cons_wrap(self, id, m, v, theta_dict):
        """ipopt wrapper for constraints

        Parameters
        ----------
        id : int
            id of government for which to calculate best response
        m : matrix
            self.N times self.N matrix of military allocations
        v : vector
            vector length self.N of governments preference parameters
        wv : vector
            vector length self.N of war values against gov id

        Returns
        -------
        function
            ipopt-suitable function to calculate constraints

        """
        def f(x, out):
            out[()] = self.Lzeros_i_cons(x, id, m, v, theta_dict)
            return(out)
        return(f)

    def Lzeros_i_cons_jac(self, xlsh, id, m, v, theta_dict):
        """calculate constraint jacobian for best response (Lagrangian)

        Parameters
        ----------
        ge_x_lbda_i_x : vector
            vector length self.x_len + self.lambda_i_len of ge vars and id's multipliers
        id : int
            id of government for which to calculate constraint jacobian
        m : matrix
            self.N times self.N matrix of military allocations
        v : vector
            vector length self.N of governments preference parameters
        theta_dict : dict (see self.rewrap_theta)
            dictionary of military parameters

        Returns
        -------
        vector
            vector of unraveled geq constraint Jacobian, Lagrange zeros Jacobian, war diffs Jacobian

        """

        jac_f = ag.jacobian(self.Lzeros_i_cons)
        out = jac_f(xlsh, id, m, v, theta_dict).ravel()

        return(out)

    def Lzeros_i_cons_jac_wrap(self, id, m, v, theta_dict):
        """wrapper around constraint Jacobian

        Parameters
        ----------
        id : int
            id of government for which to calculate constraint jacobian
        m : matrix
            self.N times self.N matrix of military allocations
        v : vector
            vector length self.N of governments preference parameters
        theta_dict : dict (see self.rewrap_theta)
            dictionary of military parameters

        Returns
        -------
        function
            ipopt-suitable function to calculate constraint jacobian

        """
        def f(x, out):
            out[()] = self.Lzeros_i_cons_jac(x, id, m, v, theta_dict)
            return(out)
        return(f)

    def Lzeros_i_bounds(self, ge_x_sv, id, bound="lower"):
        """bound input variables for best response Lagrangian calculation

        Parameters
        ----------
        ge_x_sv : vector
            vector length self.xlh_len of starting values for best response calculation
        id : int
            id of government for which to calculate best response
        bound : str
            "lower" or "upper", which bound to return

        Returns
        -------
        vector
            vector length self.x_len + self.lambda_i_len of bounds for Lagrange best response input

        """

        tau_hat = self.ecmy.rewrap_ge_dict(ge_x_sv)["tau_hat"]

        x_L = np.concatenate((np.zeros(self.x_len), np.repeat(-np.inf, self.lambda_i_len), np.zeros(self.hhat_len)))
        x_U = np.repeat(np.inf, self.L_i_len)

        tau_L = np.zeros((self.N, self.N))
        # tau_L = 1 / self.ecmy.tau
        tau_U = np.reshape(np.repeat(np.inf, self.N ** 2), (self.N, self.N))
        # tau_U = np.max(self.ecmy.tau, axis=1)
        np.fill_diagonal(tau_L, 1.)
        np.fill_diagonal(tau_U, 1.)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    if i != id:
                        # constrain others policies at starting values
                        tau_L[i, j] = tau_hat[i, j]
                        tau_U[i, j] = tau_hat[i, j]
                else:
                    tau_L[i, j] = 1.
                    tau_U[i, j] = 1.


        x_L[0:self.N**2] = tau_L.ravel()
        x_U[0:self.N**2] = tau_U.ravel()

        x_L[self.N**2:self.x_len] = self.geq_lb()[self.N**2:]  # ge lower bounds
        x_U[self.N**2:self.x_len] = self.geq_ub()[self.N**2:]  # ge upper bounds
        x_L[-self.hhat_len:] = self.geq_lb()[-self.hhat_len:]  # rc-ge lower bounds

        # x_L[self.x_len+self.lambda_i_len-self.N:self.x_len+self.lambda_i_len+self.N] = 0  # mil constraint multipliers and slack variables

        if bound == "lower":
            return(x_L)
        else:
            return(x_U)

    def dummy(self, x):
        """dummy objective function for ipopt (for when we only care about solving constraints)

        Parameters
        ----------
        x : vector
            arbitrary-lengthed vector

        Returns
        -------
        float
            constant

        """
        self.tick += 1
        if len(x) == self.xlhvt_len:
            if self.tick % 25 == 0:
                x_dict = self.rewrap_xlhvt(x)
                ge_dict = self.ecmy.rewrap_ge_dict(x_dict["ge_x"])
                lbda = np.reshape(x_dict["lbda"], (self.N, self.lambda_i_len))
                # s = np.reshape(x_dict["s"], (self.N, self.N))
                h = np.reshape(x_dict["h"], (self.N, self.hhat_len))
                v = x_dict["v"]
                print("tau:")
                print(ge_dict["tau_hat"]*self.ecmy.tau)
                print("-----")
                for i in range(self.N):
                    Lzeros_i = self.Lzeros_i_xlhvt(x, i, self.m)
                    h_diffs_i = self.h_diffs_xlhvt(x, i)
                    lbda_i = self.rewrap_lbda_i(lbda[i, ])
                    h_i = h[i, ]
                    # print("h " + str(i) + ":")
                    # print(h_i)
                    # print("h_diffs" + str(i) + ":")
                    # print(h_diffs_i)
                    # print("Lzeros:" + str(i) + ":")
                    # print(Lzeros_i)
                    print("peace probs " + str(i) + ":")
                    peace_probs_i = self.peace_probs(x_dict["ge_x"], h_i, i, self.m, x_dict["v"], self.rewrap_theta(x_dict["theta"]))
                    print(peace_probs_i)
                    print("-----")
                    # print("lbda_chi " + str(i) + ":")
                    # print(lbda_i["chi_i"])
                    # print("s " + str(i) + ":")
                    # print(s[i, ])
                    # print("-----")
                # diffs = self.estimator_cons(x, self.m)
                # print(np.argmax(diffs))
                print("ge_dict:")
                print(ge_dict)
                print("-----")
                print("G_hat:")
                print(self.G_hat(x_dict["ge_x"], x_dict["v"], 0, all=True))
                print("-----")
                print("R_hat:")
                print(self.R_hat(ge_dict, x_dict["v"]))
                print("-----")
                print("geq_diffs:")
                print(self.ecmy.geq_diffs(x_dict["ge_x"], v))
                # for i in range(self.N):
                #     rcx_i = self.rcx(ge_dict["tau_hat"], h[i, ], i)
                #     print("rcx_i:")
                #     print(self.ecmy.rewrap_ge_dict(rcx_i))
                #     print("rcv_i:")
                #     print(self.G_hat(rcx_i, x_dict["v"], 0, all=True))
                #     print("rhat_i:")
                #     print(self.R_hat(self.ecmy.rewrap_ge_dict(rcx_i), x_dict["v"]))
                #     print("-----")
        if len(x) == self.xlh_len:

            if self.tick % 25 == 0:

                id = 2
                v = (self.v_max() - 1) / 2 + 1
                theta_dict = dict()
                theta_dict["eta"] = 1
                theta_dict["alpha0"] = 0
                theta_dict["alpha1"] = .25
                theta_dict["gamma"] = 1
                theta_dict["C"] = np.repeat(.25, self.N)

                x_dict = self.rewrap_xlh(x)
                ge_dict = self.ecmy.rewrap_ge_dict(x_dict["ge_x"])
                print("tau:")
                print(ge_dict["tau_hat"]*self.ecmy.tau)
                print("-----")
                print("ge_dict:")
                print(ge_dict)
                print("-----")
                rcx_i = self.rcx(ge_dict["tau_hat"], x_dict["h"], id)
                # print("rcx_i:")
                # print(self.ecmy.rewrap_ge_dict(rcx_i))
                peace_probs = self.peace_probs(x_dict["ge_x"], x_dict["h"], id, self.m, v, theta_dict)
                print("peace probs:")
                print(peace_probs)


        c = 1
        return(c)

    def dummy_grad(self, x, out):
        """dummy objective gradient for ipopt

        Parameters
        ----------
        x : vector
            arbirtary-lengthed input vector
        out : vector
            arbirtary-lengthed out vector

        Returns
        -------
        out
            null gradient

        """
        out[()] = np.zeros(len(x))
        return(out)

    def Lsolve_i_ipopt(self, id, m, v, theta_dict):
        """solve best response (Lagrange) for i using ipopt

        Parameters
        ----------
        id : int
            government for which to calculate best response
        v : vector
            vector length self.N of governments preference parameters
        m : matrix
            self.N times self.N matrix of military allocations
        wv : vector
            vector length self.N of war values against gov id

        Returns
        -------
        vector, float, str
            ipopt outputs, solution is first

        """


        ge_x0 = self.v_sv(id, np.ones(self.x_len), v)
        # ge_x0 = np.ones(self.x_len)
        lbda_i0 = np.zeros(self.lambda_i_len)  # initialize lambdas

        ft_id = self.ecmy.rewrap_ge_dict(self.ft_sv(id, ge_x0))
        h_sv = self.ecmy.unwrap_ge_dict(self.ecmy.geq_solve(ft_id["tau_hat"], np.ones(self.N)))[-self.hhat_len:]
        # h_sv = np.ones(self.hhat_len)

        # rcx_sv = self.rcx(self.ecmy.rewrap_ge_dict(ge_x0)["tau_hat"], h_sv, id)
        # wv = self.wv_rcx(rcx_sv, id, m, v, theta_dict)
        # war_diffs = self.war_diffs(ge_x0, v, wv, id)
        # s = war_diffs

        # x0 = np.concatenate((ge_x0, lbda_i0, s, h_sv))
        x0 = np.concatenate((ge_x0, lbda_i0, h_sv))
        x_len = len(x0)

        # g_len_i = self.hhat_len + (self.hhat_len + self.N - 1) + self.N + self.N + self.hhat_len
        g_len_i = self.hhat_len + (self.hhat_len + self.N - 1) + self.hhat_len
        g_lower = np.zeros(g_len_i)
        g_upper = np.zeros(g_len_i)

        g_sparsity_indices_a = np.array(np.meshgrid(range(g_len_i), range(x_len))).T.reshape(-1,2)
        g_sparsity_indices = (g_sparsity_indices_a[:,0], g_sparsity_indices_a[:,1])
        h_sparsity_indices_a = np.array(np.meshgrid(range(x_len), range(x_len))).T.reshape(-1,2)
        h_sparsity_indices = (h_sparsity_indices_a[:,0], h_sparsity_indices_a[:,1])

        x_L = self.Lzeros_i_bounds(ge_x0, id, "lower")
        x_U = self.Lzeros_i_bounds(ge_x0, id, "upper")

        problem = ipyopt.Problem(x_len, x_L, x_U, g_len_i, g_lower, g_upper, g_sparsity_indices, h_sparsity_indices, self.dummy, self.dummy_grad, self.Lzeros_i_cons_wrap(id, m, v, theta_dict), self.Lzeros_i_cons_jac_wrap(id, m, v, theta_dict))

        problem.set(print_level=5, fixed_variable_treatment='make_parameter', linear_solver="pardiso")
        # derivative checker
        # problem.set(print_level=5, fixed_variable_treatment='make_parameter', derivative_test="first-order", point_perturbation_radius=.1)
        print("solving...")
        x_lbda, obj, status = problem.solve(x0)

        return(x_lbda, obj, status)

    def G_hat_wrap(self, v, id, sign):
        """ipopt-suitable G_hat calculator

        Parameters
        ----------
        v : vector
            vector length self.N of preference parameters
        iid : int
            id of government for which to calculate objective
        sign : float
            scalar for objective function

        Returns
        -------
        function
            ipopt-suitable objective function calculator

        """
        def f(x):
            return(self.G_hat(x, v, id, sign=sign))
        return(f)

    def G_hat_grad(self, ge_x, v, id):
        """

        Parameters
        ----------
        ge_x : vector
            vector length self.x_len of input values
        v : vector
            vector length self.N of preference parameters
        id : int
            id of government for which to calculate objective gradient

        Returns
        -------
        vector
            vector length self.x_len of gradient values

        """
        G_hat_grad_f = ag.grad(self.G_hat)
        out = G_hat_grad_f(ge_x, v, id, -1)
        return(out)

    def G_hat_grad_ipyopt_wrap(self, v, id):
        """ipopt-suitable objective function gradient

        Parameters
        ----------
        v : vector
            vector length self.N of preference parameters
        id : int
            id of government for which to calculate objective gradient

        Returns
        -------
        function
            ipopt-suitable gradient calculator

        """
        def f(ge_x, out):
            out[()] = self.G_hat_grad(ge_x, v, id)
            return(out)
        return(f)

    def br_cons_ipyopt(self, ge_x, id, v, wv):
        """calculate best response constraints (geq diffs and war diffs)

        Parameters
        ----------
        ge_x : vector
            vector length self.x_len of ge inputs
        id : int
            id of government for which to calculate best response constraints
        v : vector
            vector length self.N of preference parameters
        wv : vector
            vector length self.N of war values against gov id

        Returns
        -------
        vector
            vector length self.hhat_len + self.N of best response constraints (geq diffs and war diffs)

        """

        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        geq_diffs = self.ecmy.geq_diffs(ge_x)
        war_diffs = self.war_diffs(ge_x, v, wv, id)
        out = np.concatenate((geq_diffs, war_diffs))

        return(out)

    def br_cons_ipyopt_wrap(self, id, v, wv):
        """ipopt-suitable best response calculator

        Parameters
        ----------
        id : int
            id of government for which to calculate best response constraints
        v : vector
            vector length self.N of preference parameters
        wv : vector
            vector length self.N of war values against gov id

        Returns
        -------
        function
            ipopt-suitable function for calculating best responses

        """
        def f(x, out):
            out[()] = self.br_cons_ipyopt(x, id, v, wv)
            return(out)
        return(f)

    def br_cons_ipyopt_jac(self, ge_x, id, v, wv):
        """best response jacobian

        Parameters
        ----------
        ge_x : vector
            vector length self.x_len of ge inputs
        id : int
            id of government for which to calculate best response constraints
        v : vector
            vector length self.N of preference parameters
        wv : vector
            vector length self.N of war values against gov id

        Returns
        -------
        vector
            vector of unraveled ge diff jacobian and war diff jacobian wrt ge inputs

        """

        geq_jac_f = ag.jacobian(self.ecmy.geq_diffs)
        mat_geq = geq_jac_f(ge_x)

        wd_jac_f = ag.jacobian(self.war_diffs)
        mat_wd = wd_jac_f(ge_x, v, wv, id)
        out = np.concatenate((mat_geq.ravel(), mat_wd.ravel()))

        return(out)

    def br_cons_ipyopt_jac_wrap(self, id, v, wv):
        """ipopt-suitable best response jacobian

        Parameters
        ----------
        id : int
            id of government for which to calculate best response constraints
        v : vector
            vector length self.N of preference parameters
        wv : vector
            vector length self.N of war values against gov id

        Returns
        -------
        function
            ipopt-suitable function for calculating best response jacobian

        """
        def f(x, out):
            out[()] = self.br_cons_ipyopt_jac(x, id, v, wv)
            return(out)
        return(f)

    def br_bounds_ipyopt(self, ge_x_sv, id, bound="lower"):
        """calculate bounds for best response inputs

        Parameters
        ----------
        ge_x_sv : vector
            vector length self.x_len of best response inputs
        id : int
            id of government for which to calculate best response constraints
        bound : str
            "lower" or "upper"

        Returns
        -------
        vector
            vector length self.x_len of bounds for ipopt search

        """

        tau_hat = self.ecmy.rewrap_ge_dict(ge_x_sv)["tau_hat"]

        x_L = np.zeros(self.x_len)
        x_U = np.repeat(np.inf, self.x_len)

        tau_L = 1. / self.ecmy.tau
        tau_U = np.reshape(np.repeat(np.inf, self.N ** 2), (self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    if i != id:
                        # fix other governments' policies
                        tau_L[i, j] = tau_hat[i, j]
                        tau_U[i, j] = tau_hat[i, j]
                else:
                    # fix diagonal at 1
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

    def br_ipyopt(self, x0, v, id, wv=None):
        """calculate best response (ipopt optimizer over G_hat, enforcing ge and war constraints)

        Parameters
        ----------
        x0 : vector
            vector length self.x_len of starting values from which to optimize...fixes non-id policies at these values
        v : vector
            vector length self.N of preference parameters
        id : int
            id of government for which to calculate best response
        wv : vector
            vector length self.N of war values against gov id

        Returns
        -------
        vector
            vector length self.x_len of best response values

        """

        # constraints
        g_len = self.hhat_len + self.N
        g_upper = np.zeros(g_len)
        g_upper[-self.N:] = np.inf  # war inequalities

        g_sparsity_indices_a = np.array(np.meshgrid(range(g_len), range(self.x_len))).T.reshape(-1,2)
        g_sparsity_indices = (g_sparsity_indices_a[:,0], g_sparsity_indices_a[:,1])
        h_sparsity_indices_a = np.array(np.meshgrid(range(self.x_len), range(self.x_len))).T.reshape(-1,2)
        h_sparsity_indices = (h_sparsity_indices_a[:,0], h_sparsity_indices_a[:,1])

        # bounds
        x_L = self.br_bounds_ipyopt(x0, id, "lower")
        x_U = self.br_bounds_ipyopt(x0, id, "upper")

        problem = ipyopt.Problem(self.x_len, x_L, x_U, g_len, np.zeros(g_len), g_upper, g_sparsity_indices, h_sparsity_indices, self.G_hat_wrap(v, id, -1), self.G_hat_grad_ipyopt_wrap(v, id), self.br_cons_ipyopt_wrap(id, v, wv), self.br_cons_ipyopt_jac_wrap(id, v, wv))

        problem.set(print_level=5, nlp_scaling_method="none", fixed_variable_treatment='make_parameter')
        print("solving...")
        x, obj, status = problem.solve(x0)

        return(x)

    def rcx(self, tau_hat_tilde, h, id):
        """Short summary.

        Parameters
        ----------
        tau_hat_tilde : matrix
            matrix dim self.N**2 of current policy proposals
        h : vector
            vector length self.hhat_len of current ge response vars
        id : int
            id of government for which to construct free trade ge vars

        Returns
        -------
        vector
            vector length self.x_len of current regime change vars

        """

        tau_hat_prime = [tau_hat_tilde[j, ] if j != id else 1 / self.ecmy.tau[j, ] for j in range(self.N)]
        tau_hat_prime = np.array(tau_hat_prime)

        rcx = np.concatenate((tau_hat_prime.ravel(), np.ones(self.N), h))

        return(rcx)

    def ft_sv(self, id, ge_x, v):
        """calculate free trade equilibrium for id, holding other govs at policies in ge_x

        Parameters
        ----------
        id : int
            id of government over which to impose free trade
        ge_x : vector
            vector length self.x_len of starting values

        Returns
        -------
        vector
            vector length self.x_len of ge vars consistent with free trade for gov id, holding all other govs policies constant

        """

        ge_x_copy = np.copy(ge_x)

        tau_hat_ft = 1 / self.ecmy.tau
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x_copy)
        tau_hat_sv = ge_dict["tau_hat"]
        tau_hat_sv[id, ] = tau_hat_ft[id, ]
        # print(tau_hat_sv*self.ecmy.tau)
        ge_dict_sv = self.ecmy.geq_solve(tau_hat_sv, np.ones(self.N), v)
        if ge_dict_sv == 0:
            return(0)
        else:
            ge_x_sv = self.ecmy.unwrap_ge_dict(ge_dict_sv)
            return(ge_x_sv)

    def nft_sv(self, id, ge_x):
        """calculate near free trade values (how close defined by self.tau_nft) for government id, holding other govs at policies in ge_x

        Parameters
        ----------
        id : int
            id of government over which to impose free trade
        ge_x : vector
            vector length self.x_len of starting values

        Returns
        -------
        vector
            vector length self.x_len of ge vars consistent with near free trade for gov id, holding all other govs policies constant

        """

        tau_hat_nft = self.tau_nft / self.ecmy.tau
        np.fill_diagonal(tau_hat_nft, 1)
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        tau_hat_sv = ge_dict["tau_hat"]
        tau_hat_sv[id, ] = tau_hat_nft[id, ] # start slightly above free trade
        ge_dict_sv = self.ecmy.geq_solve(tau_hat_sv, np.ones(self.N))
        ge_x_sv = self.ecmy.unwrap_ge_dict(ge_dict_sv)

        return(ge_x_sv)

    def v_sv(self, id, ge_x, v):
        """calculate starting values consistent with goverment id setting policies at v floor

        Parameters
        ----------
        id : int
            id of government over which to impose free trade
        ge_x : vector
            vector length self.x_len of starting values
        v : vector
            vector length self.N of preference parameters

        Returns
        -------
        vector
            vector length self.x_len of ge values, holding id's policies at v_id

        """

        tau_v = np.tile(np.array([v]).transpose(), (1, self.N))
        # tau_hat_v = tau_v / self.ecmy.tau
        tau_hat_v = (tau_v + .1) / self.ecmy.tau
        np.fill_diagonal(tau_hat_v, 1)
        ge_dict = self.ecmy.rewrap_ge_dict(copy.deepcopy(ge_x))
        tau_hat_sv = ge_dict["tau_hat"]
        tau_hat_sv[id, ] = tau_hat_v[id, ] # start slightly above free trade
        ge_dict_sv = self.ecmy.geq_solve(tau_hat_sv, np.ones(self.N))
        ge_x_sv = self.ecmy.unwrap_ge_dict(ge_dict_sv)

        return(ge_x_sv)

    def v_sv_all(self, v):
        """calculate starting values imposing tau_ij = v_i for all i, j

        Parameters
        ----------
        v : vector
            vector length self.N of preference parameters

        Returns
        -------
        vector
            vector length self.x_len of starting values

        """

        step = 0

        tau_v = np.tile(np.array([v]).transpose(), (1, self.N))
        tau_hat_v = (tau_v + step) / self.ecmy.tau
        np.fill_diagonal(tau_hat_v, 1)
        ge_dict_sv = self.ecmy.geq_solve(tau_hat_v, np.ones(self.N), v)
        ge_x_sv = self.ecmy.unwrap_ge_dict(ge_dict_sv)

        return(ge_x_sv)

    def rcv_ft(self, x, v):

        out = np.zeros((self.N, self.N))
        G_hat = self.G_hat(x, v, 0, all=True)
        for i in range(self.N):
            ge_x_ft = self.ft_sv(i, x, v)
            G_prime = self.G_hat(ge_x_ft, v, i, all=True)
            out[i, ] = G_prime / G_hat

        return(out)
