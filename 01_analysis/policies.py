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
        data : dict
            dictionary storing data inputs
        params : dict
            dictionary storing economic primitive parameters
        ROWname : str
            name of rest of world aggregate ("RoW")
        results_path : string
            Path to results directory, where free trade xs will be stored

        """

        # Setup economy
        self.ecmy = economy.economy(data, params)
        self.N = self.ecmy.N
        self.ids = np.arange(self.N)

        self.ROW_id = np.where(data["ccodes"]==ROWname)[0][0]

        # purge deficits
        self.ecmy.purgeD()  # replicate DEK

        # enforce positive tariffs
        tau_hat_pos = np.ones_like(self.ecmy.tau)
        tau_hat_pos[self.ecmy.tau < 1] = 1 / self.ecmy.tau[self.ecmy.tau < 1]
        self.ecmy.update_ecmy(tau_hat_pos, np.ones(self.N))

        self.W = data["W"]  # distances
        np.fill_diagonal(self.W, 0)  # EU has positive own distance by averaging over members, zero this out
        self.M = data["M"]  # milex

        # construct equal distribution m matrix, enforce zeros on ROW row and columns
        self.m = self.M / np.ones((self.N, self.N))
        self.m = self.m.T
        self.m[self.ROW_id,:] = 0
        self.m[:,self.ROW_id] = 0
        self.m[self.ROW_id,self.ROW_id] = 1

        # counterfactual m, no threats
        self.mzeros = np.diag(self.M)

        self.tau_nft = 1.25  # near free trade policies, starting values to begin search for best response

        # len of different variable subsets
        self.hhat_len = self.N**2+4*self.N  # X, d, P, w, r, E
        self.Dhat_len = self.N
        self.tauj_len = self.N**2-self.N  # non i policies
        self.lambda_i_x_len = self.hhat_len  # ge multipliers for best response
        self.lambda_i_len = self.lambda_i_x_len + self.N  # ge multipliers plus mil multipliers

        # optimizer parameters
        self.x_len = self.ecmy.ge_x_len  # len of ge vars
        self.xlvt_len = self.x_len + self.lambda_i_len * self.N + self.N + 4  # ge vars, all lambdas (flattened), v, theta (except v)
        self.g_len = self.hhat_len + (self.hhat_len + self.N - 1)*self.N + self.N**2 + self.N**2  # constraints len: ge_diffs, Lzeros (own policies N-1), war_diffs mat, comp_slack mat
        self.max_iter_ipopt = 100000
        self.chi_min = 1.0e-10  # minimum value for chi
        self.wv_min = -1.0e2  # minimum war value
        self.alpha1_ub = self.alpha1_min(.01)  # restrict alpha search (returns alpha such that rho(alpha)=.01)
        self.zero_lb_relax = -1.0e-30  # relaxation on zero lower bound for ipopt (which are enforced without slack by ipopt (see 0.15 NLP in ipopt options))

        # generate free trade policies
        ge_x_ft_path = results_path + "ge_x_ft.csv"
        if not os.path.isfile(ge_x_ft_path):
            self.ge_x_ft = np.zeros((self.N, self.x_len))
            for i in range(self.N):
                print(str(i) + "'s free trade vector")
                ge_x_ft_i = self.ft_sv(i, np.ones(self.x_len))
                self.ge_x_ft[i, ] = ge_x_ft_i  # save i's free trade policies and ge responses in row i of matrix
            np.savetxt(ge_x_ft_path, self.ge_x_ft, delimiter=",")  # save to results_path
        else:  # load existing csv if values have already been computed
            self.ge_x_ft = np.genfromtxt(ge_x_ft_path, delimiter=",")

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
        Uhat = self.ecmy.U_hat(ge_dict)
        Ghat = Uhat * self.R_hat(ge_dict, v)

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

        return(r_hat)

    def Lagrange_i_x(self, ge_x, lambda_i_x, id, v, wv):
        """Short summary.

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function ecmy.unwrap_ge_dict for order of values.
        lambda_i_x : vector
            ge and mil multipliers for gov id
        id : int
            government choosing policy
        v : vector
            len N array of preference values for each government
        wv : vector
            Length N vector of war values for each non-id country in war against id (column id of wv matrix)

        Returns
        -------
        float
            Lagrangian evaluated at ge_x, lambda_i_x

        """

        lambda_dict_i = self.rewrap_lbda_i(lambda_i_x)
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)

        G_hat_i = self.G_hat(ge_x, v, id, sign=1)
        geq_diffs = self.ecmy.geq_diffs(ge_x)
        war_diffs = self.war_diffs(ge_x, v, wv, id)

        wd = -1 * war_diffs  # flip these so violations are positive
        # wd = np.clip(wd, 0, np.inf)

        # mil multipliers constrained to be positive, lambda_dict_i["chi_i"] > 0
        L_i = G_hat_i - np.dot(lambda_dict_i["h_hat"], geq_diffs) - np.dot(lambda_dict_i["chi_i"], wd)

        return(L_i)

    def Lzeros_i(self, ge_x_lbda_i, id, v, wv):
        """calculate first order condition for goverment i, ge_x_lbda_i input

        Parameters
        ----------
        ge_x_lbda_i : vector
            len self.x_len + self.lambda_i_len vector storing concatenated ge_x and lbda_i values
        id : int
            id of government choosing policy vector
        v: vector
            len N array of preference values for each government
        wv : vector
            len N vector of war values for each threatening government vis a vis i


        Returns
        -------
        vec
            len self.N (policies) + self.hhat_len (ge_vars) vector storing partial derivatives of government id's Lagrangian

        """

        ge_x = ge_x_lbda_i[0:self.x_len]
        lambda_i_x = ge_x_lbda_i[self.x_len:]

        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        L_grad_f = ag.grad(self.Lagrange_i_x)
        L_grad = L_grad_f(ge_x, lambda_i_x, id, v, wv)
        L_grad_i = L_grad[self.L_grad_i_ind(id)]

        out = []
        out.extend(L_grad_i)

        return(L_grad_i)

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

        # turn to zero where negative
        # wdz = np.where(war_diffs < 0, 0, war_diffs)

        return(war_diffs)

    def war_vals(self, v, m, theta_dict):
        """Calculate matrix of war values (regime change value minus war costs)

        Parameters
        ----------
        v : vector
            len self.N array of preference values for each government
        m : matrix
            self.N times self.N matrix of military allocations
        theta_dict : dict
            dictionary of military parameters (see self.rewrap_theta)

        Returns
        -------
        matrix
            self.N times self.N matrix of war values for row id in war against col id

        """

        chi = self.chi(m, theta_dict)
        chi = np.clip(chi, self.chi_min, 1)
        wc = theta_dict["c_hat"] / chi
        rcv_ft = self.rcv_ft(v)
        wv = rcv_ft - wc
        wv = np.clip(wv, self.wv_min, np.inf)  # cap minimum war values

        return(wv)

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
        rho += 1 - np.diag(np.diag(rho))  # set diagonal to one

        return(rho)

    def rcv_ft(self, v):
        """calculate matrix of G_hats, imposing free trade on each row government. (row i, column j is j's utility when i implements free trade)

        Parameters
        ----------
        v : vector
            len self.N array of preference values for each government

        Returns
        -------
        matrix
            self.N times self.N matrix of free trade values

        """

        out = np.array([self.G_hat(self.ge_x_ft[i, ], v, 0, all=True) for i in range(self.N)])

        return(out.T)

    def rewrap_xlvt(self, xlvt):
        """Convert flattened xlvt vector to dictionary

        Parameters
        ----------
        xlvt : vector (see self.unwrap_xlvt)
            len self.xlvt_len vector storing ge vars, lambdas (flattened), vs, theta

        Returns
        -------
        dict
            dictionary of flattened vectors for each sub element of xlvt

        """

        xlvt_dict = dict()
        xlvt_dict["ge_x"] = xlvt[0:self.x_len]
        xlvt_dict["lbda"] = xlvt[self.x_len:self.x_len+self.lambda_i_len*self.N]  # np.reshape(xlvt_dict["lbda"] (self.N, self.lambda_i_len)) gives matrix of lambdas, 1 row for each government
        xlvt_dict["v"] = xlvt[self.x_len+self.lambda_i_len*self.N:self.x_len+self.lambda_i_len*self.N+self.N]
        xlvt_dict["theta"] = xlvt[self.x_len+self.lambda_i_len*self.N+self.N:]

        return(xlvt_dict)

    def unwrap_xlvt(self, xlvt_dict):
        """Convert xlvt dictionary into flattened vector

        Parameters
        ----------
        xlvt_dict : dict (see self.rewrap_xlvt)
            dictionary of flattened vectors for each sub element of xlvt

        Returns
        -------
        vector
            len self.xlvt_len vector storing ge vars, lambdas (flattened), vs, theta

        """

        xlvt = []
        xlvt.extend(xlvt_dict["ge_x"])
        xlvt.extend(xlvt_dict["lbda"])
        xlvt.extend(xlvt_dict["v"])
        xlvt.extend(xlvt_dict["theta"])

        return(np.array(xlvt))

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
        theta_dict["c_hat"] = theta_x[0]
        theta_dict["gamma"] = theta_x[1]
        theta_dict["alpha0"] = theta_x[2]  # baseline power projection loss
        theta_dict["alpha1"] = theta_x[3]  # distance coefficient

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
        theta_x.extend(np.array([theta_dict["c_hat"]]))
        theta_x.extend(np.array([theta_dict["gamma"]]))
        theta_x.extend(np.array([theta_dict["alpha0"]]))
        theta_x.extend(np.array([theta_dict["alpha1"]]))

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

    def loss(self, xlvt):
        """Calculate policy loss.

        Parameters
        ----------
        xlvt : vector (see self.unwrap_xlvt)
            len self.xlvt_len vector storing ge vars, lambdas (flattened), vs, theta

        Returns
        -------
        float
            loss

        """

        ge_x = self.rewrap_xlvt(xlvt)["ge_x"]
        v = self.rewrap_xlvt(xlvt)["v"]
        theta_x = self.rewrap_xlvt(xlvt)["theta"]

        # optimizer tracker
        self.tick += 1
        if self.tick % 25 == 0:  # print output every 25 calls
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

            for i in range(self.N):
                print("lambda chi " + str(i))
                lbda = np.reshape(self.rewrap_xlvt(xlvt)["lbda"], (self.N, self.lambda_i_len))
                lbda_chi_i = self.rewrap_lbda_i(lbda[i, ])["chi_i"]
                print(lbda_chi_i)

        tau_hat = self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"]
        tau_star = tau_hat * self.ecmy.tau  # calculate equilibrium tau

        tau_diffs = tau_star - self.ecmy.tau  # difference with factual tau
        loss = np.sum(tau_diffs**2)  # euclidean norm

        return(loss)

    def loss_grad(self, xlvt, out):
        """gradient of loss function (autograd wrapper)

        Parameters
        ----------
        xlvt : vector (see self.unwrap_xlvt)
            len self.xlvt_len vector storing ge vars, lambdas (flattened), vs, theta
        out : vector
            len self.xlvt_len vector to store output (requirement for ipopt)

        Returns
        -------
        vector
            vector of partial derivatives of loss with respect to xlvt inputs

        """

        loss_grad_f = ag.grad(self.loss)
        out[()] = loss_grad_f(xlvt)

        return(out)

    def geq_diffs_xlvt(self, xlvt):
        """calculate differences between ge inputs and tau_hat-consistent ge output (ge constraints satisfied when this returns zero vector)

        Parameters
        ----------
        xlvt : vector (see self.unwrap_xlvt)
            len self.xlvt_len vector storing ge vars, lambdas (flattened), vs, theta

        Returns
        -------
        vector
            len self.h_hat_len vector of differences between ge vars in xlvt and tau_hat-consistent ge_output

        """
        ge_x = self.rewrap_xlvt(xlvt)["ge_x"]
        return(self.ecmy.geq_diffs(ge_x))

    def Lzeros_i_xlvt(self, xlvt, id, m):
        """calculate first order condition for government i, xlvt input (wrapper around self.L_zeros_i for estimation)

        Parameters
        ----------
        xlvt : vector (see self.unwrap_xlvt)
            len self.xlvt_len vector storing ge vars, lambdas (flattened), vs, theta
        id : int
            id for country for which to calculate FOC
        m : matrix
            self.N times self.N matrix of military allocations

        Returns
        -------
        vector
            len self.N (policies) + self.hhat_len (ge_vars) vector storing partial derivatives of government id's Lagrangian

        """

        xlvt_dict = self.rewrap_xlvt(xlvt)
        ge_x = xlvt_dict["ge_x"]
        lbda = np.reshape(xlvt_dict["lbda"], (self.N, self.lambda_i_len))
        lbda_i = lbda[id, ]
        v = xlvt_dict["v"]
        theta_x = xlvt_dict["theta"]
        theta_dict = self.rewrap_theta(theta_x)

        wv = self.war_vals(v, m, theta_dict)  # calculate war values

        Lzeros_i = self.Lzeros_i(np.concatenate((ge_x, lbda_i)), id, v, wv[:,id])  # flattened and concatenated ge vars and lbda_i as input

        return(Lzeros_i)

    def war_diffs_xlvt(self, xlvt, id, m):
        """calculate values of war constraints for government id, given ge vars and parameters in xlvt (wrapper around self.war_diffs)

        Parameters
        ----------
        xlvt : vector (see self.unwrap_xlvt)
            len self.xlvt_len vector storing ge vars, lambdas (flattened), vs, theta
        id : id
            id of government for which to calculate war diffs
        m : matrix
            self.N times self.N matrix of military allocations

        Returns
        -------
        vector
            length self.N vector of war diffs

        """

        xlvt_dict = self.rewrap_xlvt(xlvt)
        ge_x = xlvt_dict["ge_x"]
        v = xlvt_dict["v"]
        theta_x = xlvt_dict["theta"]
        theta_dict = self.rewrap_theta(theta_x)

        wv = self.war_vals(v, m, theta_dict)

        war_diffs_i = self.war_diffs(ge_x, v, wv[:,id], id)

        return(war_diffs_i)

    def comp_slack_xlvt(self, xlvt, id, m):
        """calculate value of complementary slackness conditions

        Parameters
        ----------
        xlvt : vector (see self.unwrap_xlvt)
            len self.xlvt_len vector storing ge vars, lambdas (flattened), vs, theta
        id : id
            id of government for which to calculate comp slack condition
        m : matrix
            self.N times self.N matrix of military allocations

        Returns
        -------
        vector
            length self.N vector of complementary slackness conditions for government id

        """

        xlvt_dict = self.rewrap_xlvt(xlvt)
        ge_x = xlvt_dict["ge_x"]
        lbda = np.reshape(xlvt_dict["lbda"], (self.N, self.lambda_i_len))
        lbda_i = lbda[id, ]
        v = xlvt_dict["v"]
        theta_x = xlvt_dict["theta"]
        theta_dict = self.rewrap_theta(theta_x)

        wv = self.war_vals(v, m, theta_dict)

        war_diffs_i = self.war_diffs(ge_x, v, wv[:,id], id)
        lbda_i_chi = self.rewrap_lbda_i(lbda_i)["chi_i"]
        comp_slack_i = war_diffs_i * lbda_i_chi

        return(comp_slack_i)

    def estimator_cons(self, xlvt, m):
        """return flattened vector of estimation constraints (length self.g_len). Equality and inequality included. Equality constraints set to zero. Inequality constraints greater than zero. These values set in self.estimator.

        Parameters
        ----------
        xlvt : vector (see self.unwrap_xlvt)
            len self.xlvt_len vector storing ge vars, lambdas (flattened), vs, theta
        m : matrix
            self.N times self.N matrix of military allocations

        Returns
        -------
        vector
            length self.g_len of estimation constraints

        """

        # geq constraints
        geq_diffs = self.geq_diffs_xlvt(xlvt)

        # Lagrange gradient
        Lzeros = []
        war_diffs = []
        comp_slack = []
        for i in range(self.N):
            Lzeros_i = self.Lzeros_i_xlvt(xlvt, i, m)
            Lzeros.extend(Lzeros_i)
            war_diffs_i = self.war_diffs_xlvt(xlvt, i, m)
            war_diffs.extend(war_diffs_i)
            comp_slack_i = self.comp_slack_xlvt(xlvt, i, m)
            comp_slack.extend(comp_slack_i)

        out = np.concatenate((geq_diffs, Lzeros, war_diffs, comp_slack), axis=None)

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

    def estimator_cons_jac(self, xlvt, g_sparsity_bin, m):
        """calculate constraint Jacobian (autograd wrapper)

        Parameters
        ----------
        xlvt : vector (see self.unwrap_xlvt)
            len self.xlvt_len vector storing ge vars, lambdas (flattened), vs, theta
        g_sparsity_bin : vector
            flattened boolean array of indices to include in return (default: all true)
        m : matrix
            self.N times self.N matrix of military allocations

        Returns
        -------
        vector
            flattened self.xlvt_len times self.g_len matrix of constraint jacobian values

        """

        geq_diffs_jac_f = ag.jacobian(self.geq_diffs_xlvt)
        geq_diffs_jac = geq_diffs_jac_f(xlvt)

        Lzeros_i_jac_f = ag.jacobian(self.Lzeros_i_xlvt)
        war_diffs_i_jac_f = ag.jacobian(self.war_diffs_xlvt)
        comp_slack_i_jac_f = ag.jacobian(self.comp_slack_xlvt)

        Lzeros_jac_flat = []
        war_diffs_jac_flat = []
        comp_slack_flat = []
        for i in range(self.N):
            Lzeros_i_jac = Lzeros_i_jac_f(xlvt, i, m)
            Lzeros_jac_flat.extend(Lzeros_i_jac.ravel())
            war_diffs_i_jac = war_diffs_i_jac_f(xlvt, i, m)
            war_diffs_jac_flat.extend(war_diffs_i_jac.ravel())
            comp_slack_i_jac = comp_slack_i_jac_f(xlvt, i, m)
            comp_slack_flat.extend(comp_slack_i_jac.ravel())

        out_full = np.concatenate((geq_diffs_jac.ravel(), Lzeros_jac_flat, war_diffs_jac_flat, comp_slack_flat), axis=None)
        out = out_full[g_sparsity_bin]

        return(out)

    def estimator_cons_jac_wrap(self, g_sparsity_bin, m):
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
            out[()] = self.estimator_cons_jac(x, g_sparsity_bin, m)
            return(out)

        return(f)

    def estimator_bounds(self, bound="lower", nash_eq=False, theta_x=None, v=None):
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
            length self.xlvt_len vector of lower or upper bounds for input values

        """

        # set initial values for bounds
        x_L = np.repeat(-np.inf, self.xlvt_len)
        x_U = np.repeat(np.inf, self.xlvt_len)

        # bound tau_hats below at zero
        tau_hat_lb = np.zeros((self.N, self.N))
        tau_hat_ub = np.max(self.ecmy.tau) / self.ecmy.tau
        # bound own tau_hats at one
        np.fill_diagonal(tau_hat_lb, 1.)
        np.fill_diagonal(tau_hat_ub, 1.)

        x_L[0:self.x_len] = 0.
        x_L[0:self.N**2] = tau_hat_lb.ravel()
        x_U[0:self.N**2] = tau_hat_ub.ravel()
        x_L[self.N**2:self.N**2+self.N] = 1.
        x_U[self.N**2:self.N**2+self.N] = 1. # fix deficits

        lbda_i_bound_dict = dict()
        lbda_i_bound_dict["h_hat"] = np.repeat(-np.inf, self.hhat_len)
        # lbda_i_bound_dict["chi_i"] = np.repeat(self.zero_lb_relax, self.N)
        lbda_i_bound_dict["chi_i"] = np.repeat(0., self.N)  # constrain inequality constraint multipliers to be positive
        lbda_i_bound = self.unwrap_lbda_i(lbda_i_bound_dict)

        lbda_bound = np.tile(lbda_i_bound, self.N)

        x_L[self.x_len:self.x_len+self.lambda_i_len*self.N] = lbda_bound  # mil constraint multipliers

        if nash_eq == False:  # set lower bounds on parameters, of fix some values for testing estimator
            x_L[self.x_len+self.lambda_i_len*self.N:self.x_len+self.lambda_i_len*self.N+self.N] = 1 # vs
            x_U[self.x_len+self.lambda_i_len*self.N:self.x_len+self.lambda_i_len*self.N+self.N] = np.max(self.ecmy.tau) # vs
            # x_L[self.x_len+self.lambda_i_len*self.N+self.N] = 0  # c_hat lower
            x_L[self.x_len+self.lambda_i_len*self.N+self.N] = .5
            x_U[self.x_len+self.lambda_i_len*self.N+self.N] = .5  # fix c_hat
            # x_L[self.x_len+self.lambda_i_len*self.N+self.N+1] = 0  # gamma lower
            # x_U[self.x_len+self.lambda_i_len*self.N+self.N+1] = 2  # gamma upper
            x_L[self.x_len+self.lambda_i_len*self.N+self.N+1] = 1
            x_U[self.x_len+self.lambda_i_len*self.N+self.N+1] = 1  # fix gamma at 1
            x_L[self.x_len+self.lambda_i_len*self.N+self.N+2] = 0  # fix alpha0
            x_U[self.x_len+self.lambda_i_len*self.N+self.N+2] = 0
            x_L[self.x_len+self.lambda_i_len*self.N+self.N+3] = -self.alpha1_ub  # alpha1 lower
            x_U[self.x_len+self.lambda_i_len*self.N+self.N+3] = self.alpha1_ub  # alpha1 upper
            # x_L[self.x_len+self.lambda_i_len*self.N+self.N+1] = 0  # alpha lower
            # x_L[self.x_len+self.lambda_i_len*self.N+self.N+1] = 0
            # x_U[self.x_len+self.lambda_i_len*self.N+self.N+1] = 0  # fix alpha
            # x_L[self.x_len+self.lambda_i_len*self.N+self.N+2] = 0  # gamma lower
        else:  # fix all parameters at initial values
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

    def estimator(self, v_sv, theta_x_sv, m, nash_eq=False):
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

        x_len = self.xlvt_len

        wd_g = np.repeat(np.inf, self.N**2)
        g_upper = np.zeros(self.g_len)
        g_upper[self.hhat_len + (self.hhat_len + self.N - 1)*self.N:self.hhat_len + (self.hhat_len + self.N - 1)*self.N+self.N**2] = wd_g

        xlvt_sv_dc = np.concatenate((np.ones(self.x_len), np.repeat(.01, self.lambda_i_len*self.N), v_sv, theta_x_sv))  # NOTE: for derivative checker, we will use these to calculate Jacobian sparsity
        xlvt_sv = np.concatenate((np.ones(self.x_len), np.zeros(self.lambda_i_len*self.N), v_sv, theta_x_sv)) # initialize starting values

        # Jacobian sparsity (none)
        g_sparsity_indices_a = np.array(np.meshgrid(range(self.g_len), range(x_len))).T.reshape(-1,2)
        g_sparsity_indices = (g_sparsity_indices_a[:,0], g_sparsity_indices_a[:,1])
        g_sparsity_bin = np.repeat(True, self.g_len*self.xlvt_len)

        # Lagrangian Hessian sparsity (none)
        h_sparsity_indices_a = np.array(np.meshgrid(range(self.xlvt_len), range(self.xlvt_len))).T.reshape(-1,2)
        h_sparsity_indices = (h_sparsity_indices_a[:,0], h_sparsity_indices_a[:,1])

        if nash_eq == False:
            b_L = self.estimator_bounds("lower")
            b_U = self.estimator_bounds("upper")
        else:
            b_L = self.estimator_bounds("lower", True, theta_x_sv, v_sv)
            b_U = self.estimator_bounds("upper", True, theta_x_sv, v_sv)

        if nash_eq == False:
            problem = ipyopt.Problem(self.xlvt_len, b_L, b_U, self.g_len, np.zeros(self.g_len), g_upper, g_sparsity_indices, h_sparsity_indices, self.loss, self.loss_grad, self.estimator_cons_wrap(m), self.estimator_cons_jac_wrap(g_sparsity_bin, m))
            problem.set(print_level=5, fixed_variable_treatment='make_parameter', max_iter=self.max_iter_ipopt, mu_strategy="adaptive")
            # for derivative test, make sure we don't travel too far from initial point with point_perturbation_radius (leads to evaluation errors)
            # problem.set(print_level=5, fixed_variable_treatment='make_parameter', max_iter=self.max_iter_ipopt, derivative_test="first-order", point_perturbation_radius=0.)
        else:
            problem = ipyopt.Problem(self.xlvt_len, b_L, b_U, self.g_len, np.zeros(self.g_len), g_upper, g_sparsity_indices, h_sparsity_indices, self.dummy, self.dummy_grad, self.estimator_cons_wrap(m), self.estimator_cons_jac_wrap(g_sparsity_bin, m))
            problem.set(print_level=5, fixed_variable_treatment='make_parameter', max_iter=self.max_iter_ipopt, linear_solver="pardiso")
        print("solving...")
        _x, obj, status = problem.solve(xlvt_sv)

        return(_x, obj, status)

    def Lzeros_i_cons(self, ge_x_lbda_i_x, id, v, wv):
        """Constraints for best response (Lagrangian version)

        Parameters
        ----------
        ge_x_lbda_i_x : vector
            vector length self.x_len + self.lambda_i_len of ge vars and id's multipliers
        id : int
            id of government for which to calculate best response
        v : vector
            vector length self.N of governments preference parameters
        wv : vector
            vector length self.N of war values against gov id

        Returns
        -------
        vector
            vector length self.hhat_len + self.lambda_i_len + self.N of geq_diffs, Lagrange zeros, and war diffs (clipped to convert to equality constraints)

        """

        ge_x = ge_x_lbda_i_x[0:self.x_len]

        lambda_i_x = ge_x_lbda_i_x[self.x_len:]

        geq_diffs = self.ecmy.geq_diffs(ge_x)
        Lzeros = self.Lzeros_i(ge_x_lbda_i_x, id, v, wv)
        war_diffs = self.war_diffs(ge_x, v, wv, id)
        comp_slack = war_diffs * self.rewrap_lbda_i(lambda_i_x)["chi_i"]
        # war_diffs = np.clip(war_diffs, -np.inf, 0)

        print(war_diffs)
        print(comp_slack)
        print(self.rewrap_lbda_i(lambda_i_x)["chi_i"])
        out = np.concatenate((geq_diffs, Lzeros, war_diffs, comp_slack))
        # out = np.concatenate((geq_diffs, Lzeros, war_diffs))

        return(out)

    def Lzeros_i_cons_wrap(self, id, v, wv):
        """ipopt wrapper for constraints

        Parameters
        ----------
        id : int
            id of government for which to calculate best response
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
            out[()] = self.Lzeros_i_cons(x, id, v, wv)
            return(out)
        return(f)

    def geq_diffs_lbda(self, ge_x_lbda_i_x):
        """wrapper around geq diffs for best response (Lagrangian version)

        Parameters
        ----------
        ge_x_lbda_i_x : vector
            vector length self.x_len + self.lambda_i_len of ge vars and id's multipliers

        Returns
        -------
        vector
            length self.hhat_len of ge violations

        """
        ge_x = ge_x_lbda_i_x[0:self.x_len]
        out = self.ecmy.geq_diffs(ge_x)
        return(out)

    def war_diffs_lbda(self, ge_x_lbda_i_x, v, wv, id):
        """wrapper around war diffs for best response (Lagrangian version)

        Parameters
        ----------
        ge_x_lbda_i_x : vector
            vector length self.x_len + self.lambda_i_len of ge vars and id's multipliers
        v : vector
            vector length self.N of governments preference parameters
        wv : vector
            vector length self.N of war values against gov id
        id : int
            id of government for which to calculate war constraints

        Returns
        -------
        vector
            length self.N vector of war constraints (clipped to convert to equality constraints)

        """

        ge_x = ge_x_lbda_i_x[0:self.x_len]
        wd = self.war_diffs(ge_x, v, wv, id)
        out = wd
        # out = np.clip(wd, -np.inf, 0)  # clip to convert to equality constraints

        return(out)

    def comp_slack_lbda(self, ge_x_lbda_i_x, v, wv, id):
        """calculate complementary slackness condition

        Parameters
        ----------
        ge_x_lbda_i_x : vector
            vector length self.x_len + self.lambda_i_len of ge vars and id's multipliers
        v : vector
            vector length self.N of governments preference parameters
        wv : vector
            vector length self.N of war values against gov id
        id : int
            id of government for which to calculate war constraints

        Returns
        -------
        vector
            length self.N vector of complementary slackness conditions

        """

        ge_x = ge_x_lbda_i_x[0:self.x_len]
        lambda_i_x = ge_x_lbda_i_x[self.x_len:]

        war_diffs = self.war_diffs(ge_x, v, wv, id)
        comp_slack = war_diffs * self.rewrap_lbda_i(lambda_i_x)["chi_i"]

        return(comp_slack)

    def Lzeros_i_cons_jac(self, ge_x_lbda_i_x, id, v, wv):
        """calculate constraint jacobian for best response (Lagrangian)

        Parameters
        ----------
        ge_x_lbda_i_x : vector
            vector length self.x_len + self.lambda_i_len of ge vars and id's multipliers
        id : int
            id of government for which to calculate constraint jacobian
        v : vector
            vector length self.N of governments preference parameters
        wv : vector
            vector length self.N of war values against gov id

        Returns
        -------
        vector
            vector of unraveled geq constraint Jacobian, Lagrange zeros Jacobian, war diffs Jacobian

        """

        geq_diffs_jac_f = ag.jacobian(self.geq_diffs_lbda)
        geq_diffs_jac_mat = geq_diffs_jac_f(ge_x_lbda_i_x)

        Lzero_diffs_jac_f = ag.jacobian(self.Lzeros_i)
        Lzero_jac_f_mat = Lzero_diffs_jac_f(ge_x_lbda_i_x, id, v, wv)

        war_diffs_jac_f = ag.jacobian(self.war_diffs_lbda)
        war_diffs_jac_mat = war_diffs_jac_f(ge_x_lbda_i_x, v, wv, id)

        comp_slack_jac_f = ag.jacobian(self.comp_slack_lbda)
        comp_slack_jac_mat = comp_slack_jac_f(ge_x_lbda_i_x, v, wv, id)

        out = np.concatenate((geq_diffs_jac_mat.ravel(), Lzero_jac_f_mat.ravel(), war_diffs_jac_mat.ravel(), comp_slack_jac_mat.ravel()))

        # out = np.concatenate((geq_diffs_jac_mat.ravel(), Lzero_jac_f_mat.ravel(), war_diffs_jac_mat.ravel()))

        return(out)

    def Lzeros_i_cons_jac_wrap(self, id, v, wv):
        """wrapper around constraint Jacobian

        Parameters
        ----------
        id : int
            id of government for which to calculate constraint jacobian
        v : vector
            vector length self.N of governments preference parameters
        wv : vector
            vector length self.N of war values against gov id

        Returns
        -------
        function
            ipopt-suitable function to calculate constraint jacobian

        """
        def f(x, out):
            out[()] = self.Lzeros_i_cons_jac(x, id, v, wv)
            return(out)
        return(f)

    def Lzeros_i_bounds(self, ge_x_sv, id, bound="lower"):
        """bound input variables for best response Lagrangian calculation

        Parameters
        ----------
        ge_x_sv : vector
            vector length self.x_len + self.lambda_i_len of starting values for best response calculation
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

        x_L = np.concatenate((np.zeros(self.x_len), np.repeat(-np.inf, self.lambda_i_len)))
        x_U = np.repeat(np.inf, self.x_len+self.lambda_i_len)

        tau_L = np.zeros((self.N, self.N))
        tau_U = np.reshape(np.repeat(np.inf, self.N ** 2), (self.N, self.N))
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

        # deficits
        x_L[self.N**2:self.N**2+self.N] = 1.
        x_U[self.N**2:self.N**2+self.N] = 1.

        x_L[-self.N:] = 0  # mil constraint multipliers

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

    def Lsolve_i_ipopt(self, id, v, wv):
        """solve best response (Lagrange) for i using ipopt

        Parameters
        ----------
        id : int
            government for which to calculate best response
        v : vector
            vector length self.N of governments preference parameters
        wv : vector
            vector length self.N of war values against gov id

        Returns
        -------
        vector, float, str
            ipopt outputs, solution is first

        """

        ge_x0 = self.v_sv(id, np.ones(self.x_len), v)
        lbda_i0 = np.zeros(self.lambda_i_len)  # initialize lambdas
        x0 = np.concatenate((ge_x0, lbda_i0))
        x_len = len(x0)

        # g_len_i = self.hhat_len + (self.hhat_len + self.N - 1) + self.N # ge constraints, gradient,  war diffs
        g_len_i = self.hhat_len + (self.hhat_len + self.N - 1) + self.N + self.N # ge constraints, gradient, war diffs, comp slack
        g_upper = np.zeros(g_len_i)
        g_upper[-self.N*2:-self.N] = np.inf

        g_sparsity_indices_a = np.array(np.meshgrid(range(g_len_i), range(x_len))).T.reshape(-1,2)
        g_sparsity_indices = (g_sparsity_indices_a[:,0], g_sparsity_indices_a[:,1])
        h_sparsity_indices_a = np.array(np.meshgrid(range(x_len), range(x_len))).T.reshape(-1,2)
        h_sparsity_indices = (h_sparsity_indices_a[:,0], h_sparsity_indices_a[:,1])

        problem = ipyopt.Problem(x_len, self.Lzeros_i_bounds(ge_x0, id, "lower"), self.Lzeros_i_bounds(ge_x0, id, "upper"), g_len_i, np.zeros(g_len_i), g_upper, g_sparsity_indices, h_sparsity_indices, self.dummy, self.dummy_grad, self.Lzeros_i_cons_wrap(id, v, wv), self.Lzeros_i_cons_jac_wrap(id, v, wv))

        problem.set(print_level=5, fixed_variable_treatment='make_parameter', linear_solver="pardiso")
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

    def ft_sv(self, id, ge_x):
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

        tau_hat_ft = 1 / self.ecmy.tau
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        tau_hat_sv = ge_dict["tau_hat"]
        tau_hat_sv[id, ] = tau_hat_ft[id, ]
        ge_dict_sv = self.ecmy.geq_solve(tau_hat_sv, np.ones(self.N))
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
        tau_hat_v = tau_v / self.ecmy.tau
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

        tau_v = np.tile(np.array([v]).transpose(), (1, self.N))
        tau_hat_v = (tau_v + .1) / self.ecmy.tau
        np.fill_diagonal(tau_hat_v, 1)
        ge_dict_sv = self.ecmy.geq_solve(tau_hat_v, np.ones(self.N))
        ge_x_sv = self.ecmy.unwrap_ge_dict(ge_dict_sv)

        return(ge_x_sv)
