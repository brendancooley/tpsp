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

        self.hhat_len = self.N**2+4*self.N
        self.tauj_len = self.N**2-self.N
        # self.lambda_i_len = self.hhat_len + self.tauj_len + 1 + self.N + (self.N - 1)  # ge vars, other policies, tau_ii, deficits, mil constraints
        self.lambda_i_len = self.hhat_len + 1 + (self.N - 1)
        # self.lambda_i_len_td = self.lambda_i_len + self.N ** 2 - self.N # add constraints on others' policies

        # NOTE: values less than zero seem to mess with best response
        # self.b_vals = np.arange(0, 1.1, .1)  # preference values for which to generate regime change value matrix.
        self.v_step = .1
        self.v_vals = np.arange(1, np.max(self.ecmy.tau), self.v_step)
        # self.v_vals = np.arange(1, 1.1, .1)
        np.savetxt(results_path + "v_vals.csv", self.v_vals, delimiter=",")

        self.x_len = self.ecmy.ge_x_len

        rcv_path = results_path + "rcv.csv"
        if not os.path.isfile(rcv_path):
            rcv = self.pop_rc_vals(rcv_ft=rcv_ft)
            self.rc_vals_to_csv(rcv, rcv_path)
            self.rcv = rcv
        else:
            self.rcv = self.read_rc_vals(rcv_path)

        self.alpha_len = 2
        # self.theta_len = self.N + self.alpha_len + 2  # b, alpha, gamma, c_hat
        self.theta_len = self.N + self.alpha_len + 1  # b, alpha, c_hat

        self.y_len = self.theta_len + self.N ** 2 + self.N + self.lambda_i_len * self.N  # parameters, military allocations, military budget multipliers, other multipliers

        self.clock = 0
        self.minute = 0

    def G_hat(self, x, v, ids=None, affinity=None, sign=1, mpec=True, jitter=True, log=False):
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
        mpec : bool
            Calculate wrt mpec (true) or enforce equilibrium in calculation (false)

        Returns
        -------
        vector
            G changes for selected governments

        """

        if affinity is None:
            affinity = np.zeros((self.N, self.N))

        # if len(x) > self.ecmy.ge_x_len:  # convert input vector to ge vars if dealing with full vector
        #     ge_x = self.rewrap_x(x)["ge_x"]
        # else:
        #     ge_x = x

        if ids is None:  # return G for all govs if no id specified
            ids = self.ids

        ge_dict = self.ecmy.rewrap_ge_dict(x)

        if mpec == False: # enforce consistency of endogenous variables with policies
            ge_dict = self.ecmy.geq_solve(ge_dict["tau_hat"], ge_dict["D_hat"])
            if jitter == True:
                if type(ge_dict) != dict:
                    ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
                    # jitter entries slightly if no solution found
                    for i in range(self.N):
                        for j in range(self.N):
                            if i != j:
                                ge_dict["tau_hat"][i, j] += np.random.normal(0, .1, 1)
                    # recurse
                    ge_x = self.ecmy.unwrap_ge_dict(ge_dict)
                    self.G_hat(ge_x, v, ids=ids, sign=sign, mpec=False, jitter=True)
                else:
                    pass

        Uhat = self.ecmy.U_hat(ge_dict)
        # Ghat = Uhat ** (1 - b) * ge_dict["r_hat"] ** b
        Ghat = Uhat * self.R_hat(ge_dict, v)

        Ghat_a = affinity * Ghat
        Ghat_out = Ghat + np.sum(Ghat_a, axis=1)

        if log==False:
            return(Ghat_out[ids]*sign)
        else:
            return(np.log(Ghat_out[ids])*sign)

    def r_v(self, ge_dict, v):

        v_mat = np.array([v])
        tau_mv = self.ecmy.tau - np.tile(v_mat.transpose(), (1, self.N))
        tau_mv[tau_mv < 0] = 0
        r = np.sum(tau_mv * self.ecmy.Xcif, axis=1)

        return(r)

    def R_hat(self, ge_dict, v):

        v_mat = np.array([v])
        r = self.r_v(ge_dict, v)

        tau_prime = ge_dict["tau_hat"] * self.ecmy.tau
        tau_prime_mv = tau_prime - np.tile(v_mat.transpose(), (1, self.N))
        tau_prime_mv[tau_prime_mv < 0] = 0
        X_prime = ge_dict["X_hat"] * self.ecmy.Xcif
        r_prime = np.sum(tau_prime_mv * X_prime, axis=1)

        r_hat = r_prime / r
        r_hat[r==0] = 0

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

    def Lagrange_i_x(self, ge_x, b, tau_hat, war_vals, lambda_i_x, id):
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

        G_hat_i = self.G_hat(ge_x, b, np.array([id]), sign=1, mpec=True, log=False)
        geq_diffs = self.ecmy.geq_diffs(ge_x)
        tau_diffs = self.tau_diffs(ge_dict["tau_hat"], tau_hat, id)
        tau_ii_diff = ge_dict["tau_hat"][id, id] - 1
        D_diffs = ge_dict["D_hat"] - 1
        war_diffs = self.war_diffs(ge_x, b, war_vals, id)

        L_i = G_hat_i - np.dot(lambda_dict_i["h_hat"], geq_diffs) - lambda_dict_i["tau_ii"] * tau_ii_diff - np.dot(lambda_dict_i["chi_i"], war_diffs)

        return(L_i)

    def war_vals(self, v, m, theta_dict, epsilon, c_bar=10):
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

        rhoM = self.rhoM(theta_dict, epsilon)
        wv = np.zeros_like(m)

        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    v_j = v[j]
                    v_j_nearest = hp.find_nearest(self.v_vals, v_j)
                    rcv_ji = self.rcv[v_j_nearest][j, i]  # get regime change value for j controlling i's policy
                    m_x = self.unwrap_m(m)
                    chi_ji = self.chi(m_x, j, i, theta_dict, rhoM)
                    # print(chi_ji)
                    if chi_ji != 0:  # calculate war costs
                        c_ji = theta_dict["c_hat"] / chi_ji
                    else:
                        c_ji = c_bar
                    wv_ji = rcv_ji - c_ji
                    wv[j, i] = wv_ji  # war value for j (row) in war against column (i)

        return(wv)

    def Lzeros(self, x, lambda_i_x, b, tau_hat, war_vals, id, bound="lower", geq=False, td=False):
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

        ge_dict = self.ecmy.rewrap_ge_dict(x)
        lambda_dict_i = self.rewrap_lambda_i(lambda_i_x)
        L_grad_f = ag.grad(self.Lagrange_i_x)
        L_grad = L_grad_f(x, b, tau_hat, war_vals, lambda_i_x, id)

        L_grad_i = L_grad[self.N*id:self.N*(id+1)]
        L_grad_h = L_grad[self.N**2+self.N:]  # skip policies and deficits
        L_grad_out = []
        L_grad_out.extend(L_grad_i)
        L_grad_out.extend(L_grad_h)
        L_grad_out = np.array(L_grad_out)
        L_grad = L_grad_out

        geq_diffs = self.ecmy.geq_diffs(x)
        # tau_diffs = self.tau_diffs(ge_dict["tau_hat"], tau_hat, id)
        tau_ii_diff = np.array([ge_dict["tau_hat"][id, id] - 1])
        # D_diffs = ge_dict["D_hat"] - 1

        # calculate war constraints
        war_diffs = self.war_diffs(x, b, war_vals, id)

        out = []
        out.extend(L_grad)
        # out.extend(np.array([np.sum(L_grad**2)]))  # sum of squared focs is zero
        if geq == True:
            out.extend(geq_diffs)
            # out.extend(D_diffs)
        # if td == True:
        #     out.extend(tau_diffs)
        out.extend(tau_ii_diff)
        out.extend(war_diffs)  # NOTE: converted these to equality constraints, where negative values turned to zeros

        # NOTE: squaring output helps keep solutions away from corners, sometimes...
        if bound == "lower":
            # return(np.array(out)**2)
            return(np.array(out))
        else:
            return(np.array(out)*-1)

    def Lzeros_tixlbda(self, tixlbda, b, tau_hat, war_vals, id, geq=False):

        th = np.copy(tau_hat)
        th[id, ] = tixlbda[0:self.N]
        gelbda = tixlbda[self.N:]

        xlbda = []
        xlbda.extend(th.ravel())  # policies
        xlbda.extend(np.ones(self.N))  # deficits
        xlbda.extend(gelbda)  # ge vars and lambdas
        xlbda = np.array(xlbda)

        x = xlbda[0:self.x_len]  # ge vars
        lbda = xlbda[self.x_len:]  # lambda_i_x

        out = self.Lzeros(x, lbda, b, tau_hat, war_vals, id, geq=geq)

        return(out)

    def war_diffs(self, ge_x, b, war_vals, id):
        """Calculate difference between government id's utility at proposed vector ge_x versus war_value

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs.
        war_vals : vector
            Length N minus one vector of war values for each non-id country in war against id
        id : int
            Gov id for which to calculate war constraints

        Returns
        -------
        vector
            Length N minus one vector of peace-war against id utility differences

        """

        ids = np.arange(self.N)
        ids = np.delete(ids, id)
        G = self.G_hat(ge_x, b, ids=ids, log=False)
        war_diffs = war_vals - G

        # turn to zero where negative
        wdz = np.where(war_diffs < 0, 0, war_diffs)

        return(wdz)

    def Lzeros_m_y(self, y, j, bound="lower"):
        """Wrapper for Lzeros_m taking flattened y vector

        Parameters
        ----------
        y : vector
            Flattened vector of parameters, military allocations, and multipliers.
        j : int
            id of government for which to calculate zeros
        bound : str
            "lower" or "upper" - which bound to return on constraint

        Returns
        -------
        vector
            zeros of j's military allocation Lagrangian

        """

        y_dict = self.rewrap_y(y)

        theta_dict = self.rewrap_theta(y_dict["theta_m"])
        lambda_m = y_dict["lambda_M"]
        lambda_dict = self.rewrap_lambda(y_dict["lambda_x"])
        m_x = y_dict["m"]

        rhoM = self.rhoM(theta_dict)

        out = self.Lzeros_m(m_x, j, lambda_m, theta_dict, rhoM, lambda_dict)

        if bound == "lower":
            return(out)
        else:
            return(-1*out)

    def Lzeros_m(self, m_x, j, lambda_m, theta_dict, rhoM, lambda_dict):
        """Calculate zeros of military allocation Lagrangian

        Parameters
        ----------
        m_x : vector
            Flattened vector of military allocations
        j : int
            id of country for which to calculate zeros
        lambda_m : vector
            Length n vector of military budget constraint multipliers
        theta_dict : dict
            Dictionary storing military structural parameters
        rhoM : matrix
            N times N symmetric matrix loss of strength gradient
        lambda_dict : dict
            Dictionary of ge, policy, and chi multipliers

        Returns
        -------
        vector
            Length N + 1 vector of optimality conditions plus (equality) constraint condition

        """

        lambda_mj = lambda_m[j]
        ids = np.delete(np.arange(0, self.N), j)
        out = []

        # m_ji
        for i in ids:
            dGdm_ji = self.dGdm_ji(m_x, j, i, theta_dict, rhoM, lambda_dict)
            out.extend(np.array([dGdm_ji - lambda_mj]))

        # m_jj
        dGdm_jj = self.dGdm_ii(m_x, j, theta_dict, rhoM, lambda_dict)
        out.extend(np.array([dGdm_jj - lambda_mj]))

        # constraint zero
        m = self.rewrap_m(m_x)
        out.extend(np.array([self.M[j] - np.sum(m[j, ])]))

        return(np.array(out))

    def dGdm_ii(self, m_x, i, theta_dict, rhoM, lambda_dict, max=100.0):
        """Calculate derivative of constrained policy Legrangian with respect to m_ii

        Parameters
        ----------
        m_x : vector
            Flattened vector of military allocations
        i : int
            id of defending country
        theta_dict : dict
            Dictionary storing military structural parameters
        rhoM : matrix
            N times N symmetric matrix loss of strength gradient
        lambda_dict : dict
            Dictionary of ge, policy, and chi multipliers
        max : float
            Value to return when derivative is undefined

        Returns
        -------
        float
            derivative

        """

        lambda_i = lambda_dict[i] # get multipliers for i
        lambda_i_dict = self.rewrap_lambda_i(lambda_i)
        ids = np.delete(np.arange(0, self.N), i)
        out = 0
        for j in range(self.N - 1):  # NOTE: these index multipliers, not govs. Use ids[j] to get corresponding id
            lambda_ij = lambda_i_dict["chi_i"][j]  # get relevant multiplier
            dChidm = ag.grad(self.chi)
            dChidm_x = dChidm(m_x, ids[j], i, theta_dict, rhoM)  # calculate gradient with respect to all allocations
            dChidm_ji = np.reshape(dChidm_x, (self.N, self.N))[i, i]  # extract relevant entry
            chi_ji = self.chi(m_x, ids[j], i, theta_dict, rhoM)
            if chi_ji != 0:
                out -= lambda_ij * theta_dict["c_hat"][0] * dChidm_ji / chi_ji ** 2
            else:
                out -= lambda_ij * max  # if chi_ji = 0 and lambda_ij != 0 return large number

        return(out)


    def dGdm_ji(self, m_x, j, i, theta_dict, rhoM, lambda_dict, max=100.0, lbda_min=.0001):
        """Return derivative of welfare with respect to military effort allocated by j against i

        Parameters
        ----------
        m_x : vector
            Flattened vector of military allocations
        j : int
            id of threatening country
        i : int
            id of defending country
        theta_dict : dict
            Dictionary storing military structural parameters
        rhoM : matrix
            N times N symmetric matrix loss of strength gradient
        lambda_dict : dict
            Dictionary of ge, policy, and chi multipliers
        max : float
            Value to return when derivative is undefined

        Returns
        -------
        float
            derivative

        """

        # TODO vectorize this and chi if possible

        # get relevant multiplier
        lambda_i = lambda_dict[i]
        lambda_i_dict = self.rewrap_lambda_i(lambda_i)
        ids = np.delete(np.arange(0, self.N), i)
        j_pos = np.where(ids == j)
        lambda_ij = lambda_i_dict["chi_i"][j_pos]

        if lambda_ij > lbda_min: # TODO: check machine precision on this
            dChidm = ag.grad(self.chi)
            dChidm_x = dChidm(m_x, j, i, theta_dict, rhoM)
            dChidm_ji = np.reshape(dChidm_x, (self.N, self.N))[j, i]
            chi_ji = self.chi(m_x, j, i, theta_dict, rhoM)
            if chi_ji != 0:
                dGdm_ji = theta_dict["c_hat"][0] * dChidm_ji / chi_ji ** 2
                return(dGdm_ji)
            else:
                return(max)
        else:
            return(0.0)

    def chi(self, m_x, j, i, theta_dict, rhoM):
        """Short summary.

        Parameters
        ----------
        m_x : vector
            flattened vector of military deployments
        j : int
            Attacking country id
        i : int
            Defending country id
        theta_dict : dict
            Dictionary storing military structural parameters
        rhoM : matrix
            N times N symmetric matrix loss of strength gradient

        Returns
        -------
        float
            probability j wins offensive war against i

        """

        # TODO: make sure this is bounded above at 1 and below at zero

        m = self.rewrap_m(m_x)

        m_ii = m[i, i]
        m_ji = m[j, i]

        # num_ji = (m_ji * rhoM[j, i])  # numerator
        # den_ji = (num_ji + m_ii)  # denominator
        num_ji = (m_ji ** theta_dict["gamma"] * rhoM[j, i])  # numerator
        den_ji = (num_ji + m_ii ** theta_dict["gamma"])  # denominator
        if den_ji != 0:
            chi_ji = num_ji / den_ji
        else:
            chi_ji = 1.

        return(chi_ji)

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

    def Lsolve(self, tau_hat, b, m, theta_dict, id, epsilon=None, ft=False, mtd="lm"):
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

        j = np.delete(np.arange(self.N), id)

        if ft is False:
            # initalize starting values to be consistent with initial tau_hat
            ge_dict_sv = self.ecmy.geq_solve(tau_hat, np.ones(self.N))
            ge_x_sv = self.ecmy.unwrap_ge_dict(ge_dict_sv)
        else:
            # initialize starting values close to free trade for gov id
            # NOTE: pure free trade locks r_hat at zero and stops search for optimum
            th = self.tau_hat_ft_i(tau_hat, id)
            th[id, ][j] += .1
            ge_dict_sv = self.ecmy.geq_solve(th, np.ones(self.N))
            ge_x_sv = self.ecmy.unwrap_ge_dict(ge_dict_sv)

        lambda_i_x_sv = np.zeros(self.lambda_i_len)

        # calculate war values
        wv = self.war_vals(b, m, theta_dict, epsilon)
        wv_i = wv[:,id][j]
        print(wv_i)

        x = []
        x.extend(ge_dict_sv["tau_hat"][id, ])
        x.extend(ge_x_sv[self.N**2+self.N:self.x_len])
        x.extend(lambda_i_x_sv)

        fct = .1  # NOTE: convergence of hybr and lm is sensitive to this value
        out = opt.root(self.Lzeros_tixlbda, x0=np.array(x), method=mtd, args=(b, tau_hat, wv_i, id, True, ), options={"factor":fct})
        if out['success'] == True:
            print("success:" + str(id))
            return(out['x'])
        else:
            print("recursing...")
            if mtd == "lm":  # first try hybr
                return(self.Lsolve(tau_hat, b, m, theta_dict, id, ft=ft, mtd="hybr"))
            else:  # otherwise start from free trade
                return(self.Lsolve(tau_hat, b, m, theta_dict, id, id, ft=True, mtd="lm"))

    def constraints_tau(self, ge_dict, tau_free, wv_i, v, ge=True, deficits=False, mil=False):
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
            return(f)

        # constrain deficits
        def con_d(ge_x, bound="lower"):
            if bound == "lower":
                con = ge_x[self.N**2:self.N**2+self.N] - 1
            else:
                con = 1 - ge_x[self.N**2:self.N**2+self.N]
            return(con)

        # build constraints
        cons = []

        # policies
        for i in np.arange(0, self.N):
            for j in np.arange(0, self.N):
                if i != tau_free:
                    cons.append({'type': 'ineq','fun': con_tau(i, j, bound="lower", bv=ge_dict["tau_hat"][i, j])})
                    cons.append({'type': 'ineq','fun': con_tau(i, j, bound="upper", bv=ge_dict["tau_hat"][i, j])})
                else:
                    if i == j:
                        cons.append({'type': 'ineq','fun': con_tau(i, j, bound="lower", bv=1)})
                        cons.append({'type': 'ineq','fun': con_tau(i, j, bound="upper", bv=1)})
                    else:
                        cons.append({'type': 'ineq','fun': con_tau(i, j, bound="lower", bv=0)})

        # deficits
        if deficits == True:
            cons.append({'type': 'ineq', 'fun': con_d, 'args':("lower",)})
            cons.append({'type': 'ineq', 'fun': con_d, 'args':("upper",)})

        # ge constraints
        if ge == True:
            cons.append({'type': 'ineq', 'fun': self.ecmy.geq_diffs, 'args':("lower",)})
            cons.append({'type': 'ineq', 'fun': self.ecmy.geq_diffs, 'args':("upper",)})

        # mil constraints
        if mil == True:
            ids_j = np.delete(np.arange(self.N), tau_free)
            for j in range(self.N):
                if j != tau_free:
                    idx = np.where(ids_j==j)[0]
                    cons.append({'type': 'ineq', 'fun': self.con_mil, 'args':(tau_free, j, wv_i[idx], v, )})

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
        G_j = self.G_hat(ge_x, v, ids=np.array([j]), log=False)

        cons = G_j - wv_ji

        return(cons)

    def constraints_m(self, m_init, id):
        """Compile constraints for military strategy problem

        Parameters
        ----------
        m_init : matrix
            N times N matrix of initial military deployments
        id : int
            id of government optimizing military strategy

        Returns
        -------
        list
            List of constraints for optimizers

        """

        cons = []

        # military matrix (others' strategies fixed)
        def con_m(x, m_init, id, bound="lower"):

            # constrained ids
            ids = np.arange(self.N)
            ids = np.delete(ids, id)

            # get proposed allocations
            m_x = self.rewrap_x(x)["m_x"]
            m = self.rewrap_m(m_x)
            m_j = m[ids, ].flatten()  # constrained rows

            out = m_j - m_init[ids, ].flatten() # diffs between proposed and initial allocations

            if bound == "lower":
                return(out)
            else:
                return(out*-1)

        cons.append({'type': 'ineq', 'fun': con_m, 'args':(m_init, id, "lower",)})
        cons.append({'type': 'ineq', 'fun': con_m, 'args':(m_init, id, "upper",)})

        # military budget constraint
        def con_M(x, id):
            m_x = self.rewrap_x(x)["m_x"]
            m = self.rewrap_m(m_x)
            m_i = m[id, ]
            return(self.M[id] - np.sum(m_i))

        cons.append({'type': 'ineq', 'fun': con_M, 'args':(id, )})

        # tau optimality (Lagrange zeros)
        # NOTE: by far the most computationally intensive, play with self.war_vals if this is inefficient
        def con_L(x, bound="lower"):
            ge_x = self.rewrap_x(x)["ge_x"]
            ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
            lambda_x = self.rewrap_x(x)["lambda_x"]
            m = self.rewrap_m(self.rewrap_x(x)["m_x"])
            tau_hat = ge_dict["tau_hat"]  # current policies
            z = []
            for i in range(self.N):
                war_vals = self.war_vals(ge_dict, m, i)
                lambda_i_x = self.rewrap_lambda(lambda_x)[i]
                L_i_x = np.append(ge_x, lambda_i_x)
                z.extend(self.Lzeros(L_i_x, tau_hat, war_vals, i, bound=bound))
            if bound == "lower":
                return(np.array(z))
            else:
                return(-1 * np.array(z))

        cons.append({'type': 'ineq', 'fun': con_L, 'args':("lower",)})
        cons.append({'type': 'ineq', 'fun': con_L, 'args':("upper",)})

        return(cons)

    def br_war_ji(self, ge_x, v, j, i, mpec=True, full_opt=False):
        """puppet policies implemented by j after war on i

        Parameters
        ----------
        ge_x : vector
            Flattened array of ge inputs and outputs
        j : int
            id of coercer
        i : int
            id of defender
        mpec : type
            Description of parameter `mpec`.

        Returns
        -------
        vector
            Flattened array of ge inputs and outputs at optimal values for j choosing i's policies


        NOTE: seems a lot more stable after updating scipy to 1.3.2
        """

        # ge_x = np.copy(ge_x)
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)

        tau_perturb = .1
        v_perturb = .01

        if full_opt == True:
            # initialize starting values of ge_x to equilibrium
            # ge_dict = self.ecmy.geq_solve(ge_dict["tau_hat"], ge_dict["D_hat"])
            # ge_x = self.ecmy.unwrap_ge_dict(ge_dict)

            mxit = 500
            ftol = 1e-06
            wv_null = np.repeat(0, self.N - 1)
            cons = self.constraints_tau(ge_dict, i, wv_null, v, mil=False)
            bnds = self.bounds()
            thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, args=(v, np.array([j]), None, -1, True, ), method="SLSQP", options={"maxiter":mxit, "ftol":ftol})
            while thistar['success'] == False:
                print("iterating...")
                for k in range(self.N):
                    if k != i:
                        ge_dict["tau_hat"][k, i] = ge_dict["tau_hat"][k, i] + np.random.normal(loc=0, scale=tau_perturb)
                # v[j] -= v_perturb
                ge_dict = self.ecmy.geq_solve(ge_dict["tau_hat"], ge_dict["D_hat"])
                ge_x = self.ecmy.unwrap_ge_dict(ge_dict)
                thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, args=(v, np.array([j]), None, -1, True, ), method="SLSQP", options={"maxiter":mxit, "ftol":ftol})

            return(thistar['x'])
        else:
            print("open...")
            # "open" tariff changes
            for k in range(self.N):
                if k != i:
                    ge_dict["tau_hat"][i, k] = 1 / self.ecmy.tau[i, k]

            ge_dict_open = self.ecmy.geq_solve(ge_dict["tau_hat"], ge_dict["D_hat"])
            ge_x_open = self.ecmy.unwrap_ge_dict(ge_dict_open)
            G_hat_open = self.G_hat(ge_x_open, b, ids=np.array([j]), sign=1, mpec=True)

            print("closed...")
            # "closed" tariff changes
            for k in range(self.N):
                if k not in [i, j]:
                    ge_dict["tau_hat"][i, k] = self.tauMax / self.ecmy.tau[i, k]
                else:
                    ge_dict["tau_hat"][i, k] = 1 / self.ecmy.tau[i, k]

            ge_dict_closed = self.ecmy.geq_solve(ge_dict["tau_hat"], ge_dict["D_hat"])
            ge_x_closed = self.ecmy.unwrap_ge_dict(ge_dict_closed)
            G_hat_closed = self.G_hat(ge_x_closed, b, ids=np.array([j]), sign=1, mpec=True)

            if G_hat_open >= G_hat_closed:
                return(ge_x_open)
            else:
                return(ge_x_closed)

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
            Length N minus 1 vector of war values
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
            thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, args=(v, np.array([id]), None, -1, True, True, True, ), method="SLSQP", options={"maxiter":mxit})
        else:
            thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, args=(v, np.array([id]), affinity, -1, True, True, True, ), method="SLSQP", options={"maxiter":mxit})

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
            thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, args=(v, np.array([id]), affinity, -1, True, True, True, ), method="SLSQP", options={"maxiter":mxit})
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
            print(ge_dict)
            ge_x = self.ecmy.unwrap_ge_dict(ge_dict)
            thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, args=(v, np.array([id]), affinity, -1, True, True, True, ), method="SLSQP", options={"maxiter":mxit})
            thistar_dict = self.ecmy.rewrap_ge_dict(thistar['x'])
            taustar = thistar_dict["tau_hat"]*self.ecmy.tau

        # else:
        #     cons = self.constraints_tau(ge_dict, id, ge=False, mil=True)
        #     bnds = self.bounds()
        #     thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, args=(self.b, np.array([id]), -1, False, ), method="SLSQP", options={"maxiter":mxit})

        return(thistar['x'])

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
        tau_hat_v = tau_v / self.ecmy.tau + .001
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
        print("tau_star: " + str(tau_star))
        print("tau: " + str(self.ecmy.tau[id, ]))

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

            print("vmin: " + str(vmin))
            print("vmax: " + str(vmax))

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

            print("lb:")
            print(lb)
            print("ub:")
            print(ub)

            v = self.v_vals[idx_first]

            Loss = []  # store values for local loss
            for idx in [idx_down, idx_first, idx_up]:
                v_idx = self.v_vals[idx]
                v_vec[id] = v_idx
                print("v_vec:" + str(v_vec))
                wv = self.war_vals(v_vec, m, theta_dict, epsilon) # calculate war values
                ids_j = np.delete(np.arange(self.N), id)
                wv_i = wv[:,id][ids_j]
                # print("wv_i: " + str(wv_i))

                ge_x_sv = self.v_sv(id, np.ones(self.x_len), v_vec)

                print("v_idx: " + str(v_idx))
                br = self.br(ge_x_sv, v_vec, wv_i, id)  # calculate best response
                br_dict = self.ecmy.rewrap_ge_dict(br)
                tau_i = br_dict["tau_hat"][id, ]
                print(tau_i)
                # loss = self.loss_tau(tau_i, id, weights=self.ecmy.Y)
                loss = self.loss_tau(tau_i, id)
                Loss.append(loss)

            print("Loss: " + str(Loss))
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
                print("id: " + str(id))
                v_star = self.est_v_i_grid(id, v_vec, m, theta_dict, epsilon)
                v_vec[id] = v_star  # update vector
                print("b_vec: " + str(v_vec))
            # print("b_vec: " + str(b_vec))
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
        # print("rcv: " + str(rcv))

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

    def trunc_epsilon(self, epsilon_star, theta_dict):

        return(hp.mean_truncnorm(epsilon_star, theta_dict["sigma_epsilon"]))

    def est_theta(self, b, m, theta_dict, thres=.01):
        """Estimate military parameters from constraints. Iteratively recalculate parameters and weights until convergence.

        Parameters
        ----------
        b : vector
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

        m_diag = np.diagonal(m)
        m_frac = m / m_diag

        rcv = np.zeros((self.N, self.N))  # empty regime change value matrix (row's value for invading column)
        for i in range(self.N):
            b_nearest = hp.find_nearest(self.b_vals, b[i])
            rcv[i, ] = self.rcv[b_nearest][i, ]  # grab rcvs associated with b_nearest and extract ith row
            # (i's value for invading all others)
        # rcv = rcv.T
        print("rcv: ")
        print(rcv)

        diffs = 10
        k = 1
        while diffs > thres:
            print("k: " + str(k))
            theta_dict_last = copy.deepcopy(theta_dict)

            epsilon_star = self.epsilon_star(b, m, theta_dict, self.W)
            print(epsilon_star)
            weights = self.weights(epsilon_star, theta_dict["sigma_epsilon"])
            print(weights)
            # NOTE: weights are affected by values of theta_dict, iterate on this until convergence

            # chat = .2
            # lhs = np.log(m_frac) - np.log( 1 / (chat ** -1 * (rcv - 1) - 1) )
            lhs = np.log( 1 / (theta_dict["c_hat"] ** -1 * (rcv - 1) - 1) )
            lhs = np.nan_to_num(lhs)
            print("lhs")
            Y = lhs.ravel()
            X = np.column_stack((np.log(m_frac.ravel()), self.W.ravel()))
            X[:,0][X[:,0]==-np.inf] = np.nan
            print(X)

            # print("regressors: " + str(X))
            # print("lhs: " + str(Y))

            ests = sm.WLS(Y, X, weights=weights.ravel(), missing='drop').fit()

            theta_dict["gamma"] = ests.params[0]
            theta_dict["alpha"] = -ests.params[1]
            # theta_dict["sigma_epsilon"] = np.dot(ests.resid ** 2, weights.ravel())
            if theta_dict["gamma"] < 0:
                theta_dict["gamma"] = .01
            # if theta_dict["alpha"] < 0:
            #     theta_dict["alpha"] = .01

            theta_k = np.array([i for i in theta_dict.values()])
            theta_km1 = np.array([i for i in theta_dict_last.values()])

            diffs = np.sum((theta_k - theta_km1) ** 2)
            k += 1

        # alpha = ests.params()
        # sigma_epsilon = np.dot(ests.resid ** 2, weights.ravel())

        return(theta_dict)

    def est_loop(self, b_init, theta_dict_init, thres=.25, est_c=False, c_step=.1, c_min=.15, P=1, epsilon_zeros=True, estimates_path=""):
        """Estimate model. For each trial c_hat, iterate over estimates of b and alpha, gamma until convergence. Choose c_hat and associated parameters with lowest loss on predicted policies.

        Parameters
        ----------
        b_init : vector
            N times 1 vector of starting values for preference parameters
        theta_dict_init : dict
            Dictionary of starting values for gamma, alpha
        thres : float
            Convergence criterion, stop iteration when sum of squared changes in parameter values is less than thres
        est_c : bool
            if True then search over vector of possible c_hat values, if not then fix at initial value
        c_step : float
            step size for c estimation
        P : int
            number of epsilons to draw
        epsilon_zeros : bool
            force epsilon to zero

        Returns
        -------
        dict
            Estimates for preference parameters, military parameters, and war costs (respectively)

        """

        m = self.M / np.ones((self.N, self.N))
        m = m.T
        m[self.ROW_id,:] = 0
        m[:,self.ROW_id] = 0
        m[self.ROW_id,self.ROW_id] = 1
        print("m_frac:")
        print(m)

        if est_c is True:
            c_hat_vec = np.arange(c_min, 1 + c_step, c_step)
        else:
            c_hat_vec = [theta_dict_init["c_hat"]]
        np.savetxt(estimates_path + "c_hat_vec.csv", c_hat_vec, delimiter=",")

        Loss = []
        b = []
        alpha = []
        gamma = []

        tick = 0
        for c_hat in c_hat_vec:

            theta_dict_init["c_hat"] = c_hat

            Loss_c = []
            b_c = []
            alpha_c = []
            gamma_c = []

            epsilon = []
            for p in range(P):
                if epsilon_zeros is True:
                    epsilon_p = np.zeros((self.N, self.N))
                else:
                    epsilon_p = np.reshape(np.random.normal(0, theta_dict_k["sigma_epsilon"], self.N ** 2), (self.N, self.N))
                epsilon.append(epsilon_p)

            # pool = mp.Pool(mp.cpu_count())
            # for e in epsilon:
            #     pool.apply_async(self.est_loop_interior, args=(e, b_init, theta_dict_init, m, b_c, alpha_c, gamma_c, Loss_c))
            # pool.close()
            # pool.join()
            for e in epsilon:
                self.est_loop_interior(e, b_init, theta_dict_init, m, b_c, alpha_c, gamma_c, Loss_c)

            print(Loss_c)
            print(b_c)
            print(alpha_c)
            print(gamma_c)

            b.append(np.mean(b_c, axis=0))
            alpha.append(np.mean(alpha_c))
            gamma.append(np.mean(gamma_c))
            Loss.append(np.mean(Loss_c))

            out_dict_c = {"alpha":alpha_c[0], "gamma":gamma_c[0], "c_hat":c_hat, "sigma_epsilon":theta_dict_init["sigma_epsilon"], "Loss":Loss_c[0]}
            for id in range(self.N):
                out_dict_c["b" + str(id)] = b_c[0][id]

            self.export_results(out_dict_c, estimates_path + "ests_" + str(tick) + ".csv")
            tick += 1


        out_id = np.argmin(Loss)

        out_dict = {"alpha":alpha[out_id], "gamma":gamma[out_id], "c_hat":c_hat_vec[out_id], "sigma_epsilon":theta_dict_init["sigma_epsilon"]}
        for id in range(self.N):
            out_dict["b" + str(id)] = b[out_id][id]

        self.export_results(out_dict, estimates_path + "ests_" + "min" + ".csv")

        return(out_dict)

    def est_loop_interior(self, epsilon, b_init, theta_dict_init, m, b, alpha, gamma, Loss, thres=.01):
        """For fixed values of epsilon and c_hat, estimate preference parameters and alpha, gamma

        Parameters
        ----------
        b_init : vector
            N times 1 vector of initial preference parameters
        theta_dict_init : dict
            Dictionary storing values of alpha, gamma, c_hat, sigma_epsilon
        m : matrix
            N times N matrix of military deployments
        epsilon : matrix
            N times N matrix of war shocks
        b : list
            List to append values for b estimates
        alpha : list
            List to append values for alpha estimates
        gamma : list
            List to append values for alpha estimates
        Loss : list
            List to append values for empirical loss
        thres : float
            Convergence criterion for inner loop

        """

        b_k = np.copy(b_init)
        theta_dict_k = copy.deepcopy(theta_dict_init)

        diffs = 10
        k = 1
        while diffs > thres:

            print("k: " + str(k))

            b_km1 = np.copy(b_k)
            theta_km1 = np.copy(np.array([i for i in theta_dict_k.values()]))
            vals_km1 = np.append(b_km1, theta_km1)

            b_k = self.est_b_grid(b_k, m, theta_dict_k, epsilon)
            theta_dict_k = self.est_theta(b_k, m, theta_dict_k)

            print("b_k: " + str(b_k))
            print("theta_dict_k: " + str(theta_dict_k))

            theta_k = np.array([i for i in theta_dict_k.values()])
            vals_k = np.append(b_k, theta_k)

            print("vals_km1: " + str(vals_km1))
            print("vals_k: " + str(vals_k))
            diffs = np.sum((vals_k - vals_km1) ** 2)
            k += 1

        b.append(b_k)
        alpha.append(theta_dict_k["alpha"])
        gamma.append(theta_dict_k["gamma"])

        Loss_k = 0
        for id in range(self.N):

            # war values
            wv = self.war_vals(b_k, m, theta_dict_k, np.zeros((self.N, self.N))) # calculate war values
            ids_j = np.delete(np.arange(self.N), id)
            wv_i = wv[:,id][ids_j]

            # starting values
            ge_x_sv = self.nft_sv(id, np.ones(self.x_len))

            br = self.br(ge_x_sv, b_k, wv_i, id)  # calculate best response
            br_dict = self.ecmy.rewrap_ge_dict(br)
            tau_i = br_dict["tau_hat"][id, ]
            # Loss_k += self.loss_tau(tau_i, id, weights=self.ecmy.Y)
            Loss_k += self.loss_tau(tau_i, id)

        Loss.append(Loss_k)

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

    def affinity_cor(self, affinity_sv, b, theta_dict, m, epsilon, step=.1, delta=.05):

        # NOTE: these are difficult to calibrate exactly because constraints will pin tau_hats away from one systematically.

        affinity = np.copy(affinity_sv)

        wv = self.war_vals(b, m, theta_dict, epsilon) # calculate war values
        for id in range(self.N):
            print("id: " + str(id))
            ids_j = np.delete(np.arange(self.N), id)
            wv_i = wv[:,id][ids_j]
            ge_x_nft = self.nft_sv(id, np.ones(self.x_len))
            ge_x = self.br(ge_x_nft, b, wv_i, id, affinity=affinity)
            tau_hat_i = self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"][id, ]
            print(tau_hat_i)
            for j in range(len(tau_hat_i)):
                diff = tau_hat_i[j] - 1
                if diff > delta: # overpredicting tau, up affinity
                    affinity[id, j] += step
                if -1 * diff > delta:  # underpredictig tau, lower affinity
                    affinity[id, j] -= step
            print(affinity)

        return(affinity)

    def affinity_fp(self, b, theta_dict, m, step=.1, range = .05):

        epsilon = np.zeros((self.N, self.N))
        affinity_sv = np.zeros((self.N, self.N))
        affinity_out = opt.fixed_point(self.affinity_cor, affinity_sv, args=(b, theta_dict, m, epsilon, step, range, ), method="iteration")

        return(affinity_out)

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
        print("start:")
        print(ge_dict_sv)
        tau_hat_sv = np.copy(ge_dict_sv["tau_hat"])
        ge_x = np.copy(ge_x_sv)
        tau_hat_nft = self.tau_nft / self.ecmy.tau
        np.fill_diagonal(tau_hat_nft, 1)
        # print("nft:")
        # print(tau_hat_nft)

        tau_hat_br = ge_dict_sv["tau_hat"]
        wv = self.war_vals(b, m, theta_dict, epsilon) # calculate war values
        for id in range(self.N):
            print("id: " + str(id))
            ids_j = np.delete(np.arange(self.N), id)
            wv_i = wv[:,id][ids_j]
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

        print(tau_hat_br)
        ge_dict = self.ecmy.geq_solve(tau_hat_br, np.ones(self.N))
        ge_x = self.ecmy.unwrap_ge_dict(ge_dict)
        print("end: ")
        print(ge_dict)
        print(ge_dict["tau_hat"] - tau_hat_sv)

        print("taustar: ")
        print(ge_dict["tau_hat"]*self.ecmy.tau)

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
        lambda_dict_i["tau_ii"] = x[self.hhat_len:self.hhat_len+1]  # own policy contraint
        lambda_dict_i["chi_i"] = x[self.hhat_len+1:self.lambda_i_len]  # mil constraints, threats against i

        return(lambda_dict_i)

    def unwrap_lambda(self, lambda_dict):
        """Take nested dictionary of multipliers for each government, unwrap each and output flattened vector of all multipliers

        Parameters
        ----------
        lambda_dict : dict
            Dictionary storing dictionaries of multipliers for each government

        Returns
        -------
        vector
            Vector length N times lambda_len

        """

        x = []
        for i in range(self.N):
            x.extend(lambda_dict[i])

        return(np.array(x))

    def rewrap_lambda(self, x):
        """Convert flattened vector of all mutlipliers into dictionary storing vectors of indexed multipliers for each gov.

        Parameters
        ----------
        x : vector
            Vector length N times lambda_len.

        Returns
        -------
        dict
            Dictionary storing vectors of multipliers for each government

        """

        lambda_dict = dict()
        for i in range(self.N):
            lambda_dict[i] = x[i*self.lambda_i_len:(i+1)*self.lambda_i_len]

        return(lambda_dict)

    def tau_hat_ft_i(self, tau_hat, ids):
        """returns free-trade equivalent tau_hats for gov id, holding other policies at values in tau_hat

        Parameters
        ----------
        tau_hat : matrix
            N times N matrix of starting tau_hats
        ids : array
            Gov ids for which to change tau_hat values to free trade equivalents

        Returns
        -------
        matrix
            N times N matrix of tau_hats, with id's consistent with free trade

        """
        thfti = np.copy(tau_hat)
        thfti[ids, ] = 1 / self.ecmy.tau[ids, ]
        return(thfti)

    def pop_rc_vals(self, rcv_ft=False):
        """Generate regime change value matrix for each value of preference parameters in self.b_vals

        Returns
        -------
        dict
            Dictionary mapping b vals to to matrices of war values. Each entry is war value for row in war against column

        """

        wv = dict()
        ge_x = np.ones(self.ecmy.ge_x_len)
        for v in self.v_vals:
            print("v: " + str(v))
            # v_vec = np.repeat(v, self.N)
            wvb = np.zeros_like(self.ecmy.tau)
            for i in range(self.N):
                tau_hat_ft = 1 / self.ecmy.tau
                tau_hat_prime = np.ones((self.N, self.N))
                tau_hat_prime[i, ] = tau_hat_ft[i, ]
                ge_dict_prime = self.ecmy.geq_solve(tau_hat_prime, np.ones(self.N))
                ge_x_prime = self.ecmy.unwrap_ge_dict(ge_dict_prime)
                for j in range(self.N):
                    v_vec = np.ones(self.N)
                    v_vec[j] = v
                    if i != j:
                        print(str(j) + "'s value for replacing " + str(i))
                        if v > (np.max(self.ecmy.tau[j, ]) - self.v_step):
                            print("NA")
                            wvb[j, i] = np.NaN
                        else:
                            # start_time = time.time()
                            # populates matrix column-wise, value for row of controlling policy in column
                            if rcv_ft is False:
                                nft_sv = self.nft_sv(i, np.ones(self.x_len))
                                ge_br_war_ji = self.br_war_ji(nft_sv, v_vec, j, i, full_opt=True)
                                G_hat_ji = self.G_hat(ge_br_war_ji, v_vec, ids=np.array([j]))
                            else:
                                G_hat_ji = self.G_hat(ge_x_prime, v_vec, ids=np.array([j]))
                            wvb[j, i] = G_hat_ji
                            # print(time.time() - start_time)
            wv[v] = wvb
            print(wvb)
        return(wv)

    def rc_vals_to_csv(self, wv, fname):
        """Write war value matrices to csv. One line for each flattened matrix.

        Parameters
        ----------
        wv : dict
            dictionary storing matrices of war values for each value of preference parameters in self.b_vals
        fname : string
            file name for output csv

        """
        with open(fname, "w") as file:
            writer = csv.writer(file)
            for i in wv.keys():
                row = wv[i].flatten()
                writer.writerow(row)

    def read_rc_vals(self, fname):
        """Read war values from csv

        Parameters
        ----------
        fname : string
            Name fo file storing war values

        Returns
        -------
        dict
            Dictionary mapping b vals to to matrices of war values. Each entry is war value for row in war against column

        """
        wv = dict()
        with open(fname) as file:
            reader = csv.reader(file, delimiter=",")
            tick = 0
            for row in reader:
                v = self.v_vals[tick]
                vals = [float(i) for i in row]
                wv[v] = np.reshape(np.array(vals), (self.N, self.N))
                tick += 1
        return(wv)
