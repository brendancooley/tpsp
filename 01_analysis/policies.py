import autograd as ag
import autograd.numpy as np
import scipy.optimize as opt
import economy
import csv
import helpers
import time
import os
# import nlopt # NOTE: something wrong with build of this on laptop

# TODO: remove b from theta dict, need to track preferences separately

class policies:

    def __init__(self, data, params, b, rcv_path=None):
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

        # purge deficits
        self.ecmy.purgeD()

        # enforce positive tariffs
        tau_hat_pos = np.ones_like(self.ecmy.tau)
        tau_hat_pos[self.ecmy.tau < 1] = 1 / self.ecmy.tau[self.ecmy.tau < 1]
        self.ecmy.update_ecmy(tau_hat_pos, np.ones(self.N))

        # preference parameters
        self.b = b

        # power projection parameters
        self.alpha_0 = params["alpha_0"]
        self.alpha_1 = params["alpha_1"]
        self.gamma = params["gamma"]
        self.c_hat = params["c_hat"]

        self.W = data["W"]  # distances
        self.M = data["M"]  # milex
        # self.rhoM = self.rho()  # loss of strength gradient

        self.tauMin = 1  # enforce lower bound on policies
        self.tauMax = 20

        self.hhat_len = self.N**2+4*self.N
        self.tauj_len = self.N**2-self.N
        # self.lambda_i_len = self.hhat_len + self.tauj_len + 1 + self.N + (self.N - 1)  # ge vars, other policies, tau_ii, deficits, mil constraints
        self.lambda_i_len = self.hhat_len + 1 + (self.N - 1)
        # self.lambda_i_len_td = self.lambda_i_len + self.N ** 2 - self.N # add constraints on others' policies

        self.b_vals = np.arange(0, 1.1, .1)  # preference values for which to generate regime change value matrix. NOTE: currently only 0 and 1
        if not os.path.isfile(rcv_path):
            rcv = self.pop_rc_vals()
            self.rc_vals_to_csv(rcv, rcv_path)
            self.rcv = rcv
        else:
            self.rcv = self.read_rc_vals(rcv_path)

        self.alpha_len = 2
        # self.theta_len = self.N + self.alpha_len + 2  # b, alpha, gamma, c_hat
        self.theta_len = self.N + self.alpha_len + 1  # b, alpha, c_hat

        self.x_len = self.ecmy.ge_x_len
        self.y_len = self.theta_len + self.N ** 2 + self.N + self.lambda_i_len * self.N  # parameters, military allocations, military budget multipliers, other multipliers

        self.clock = 0
        self.minute = 0

    def G_hat(self, x, b, ids=None, sign=1, mpec=True, jitter=True, log=False):
        """Calculate changes in government welfare given ge inputs and outputs

        Parameters
        ----------
        ge_x : vector (TODO: documentation for arbitrary input vector)
            1d numpy array storing flattened ge inputs and outputs. See function for order of values.
        b : vector
            Length N vector of government preference parameters.
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
                    self.G_hat(ge_x, b, ids=ids, sign=sign, mpec=False, jitter=True)
                else:
                    pass

        Uhat = self.ecmy.U_hat(ge_dict)
        Ghat = Uhat ** (1 - b) * ge_dict["r_hat"] ** b

        if log==False:
            return(Ghat[ids]*sign)
        else:
            return(np.log(Ghat[ids])*sign)

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

    def war_vals(self, b, m, theta_dict, epsilon, c_bar=10):
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
                    b_j = b[j]
                    b_j_nearest = helpers.find_nearest(self.b_vals, b_j)
                    rcv_ji = self.rcv[b_j_nearest][j, i]  # get regime change value for j controlling i's policy
                    m_x = self.unwrap_m(m)
                    chi_ji = self.chi(m_x, j, i, theta_dict, rhoM)
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

    def constraint_xy_tau_i(self, xy_tau, id, wv_i, geq=False):

        dict_xy_tau = self.rewrap_xy_tau(xy_tau)

        x = dict_xy_tau["ge_x"]
        b = dict_xy_tau["b"]
        xnt = x[self.N**2+self.N:]  # just ge vars associated w/ geq_diffs (no taus, Ds)

        th = self.ecmy.rewrap_ge_dict(x)["tau_hat"]
        th_i = th[id, ]  #id's policies

        lambda_x = dict_xy_tau["lambda_x"]
        lambda_x_dict = self.rewrap_lambda(lambda_x)
        lambda_i_x = lambda_x_dict[id]

        input = []
        input.extend(th_i)
        input.extend(xnt)
        input.extend(lambda_i_x)
        input = np.array(input)

        out = self.Lzeros_tixlbda(input, b, th, wv_i, id, geq=geq)

        # if (time.time() - self.clock) > 10:
        #     print("dict:")
        #     print(dict_xy_tau)
        #     print("minute:" + str(self.minute))
        #     self.clock = time.time()
        #     self.minute += 1

        return(out)

    def constraint_xy_geq(self, xy_tau, bound="lower"):

        dict_xy_tau = self.rewrap_xy_tau(xy_tau)
        x = dict_xy_tau['ge_x']
        geq_diffs = self.ecmy.geq_diffs(x)
        # D_diffs = self.ecmy.rewrap_ge_dict(x)["D_hat"] - 1

        out = []
        out.extend(geq_diffs)
        # out.extend(D_diffs) # NOTE: don't need these because D fixed at zero in class pecmy
        out = np.array(out)

        if bound == "lower":
            return(out)
        else:
            return(-1*out)

    def constraint_xy_tau(self, xy_tau, m, alpha, c_hat, bound="lower"):

        theta_dict = dict()
        theta_dict["b"] = self.rewrap_xy_tau(xy_tau)["b"]
        theta_dict["alpha"] = alpha
        theta_dict["c_hat"] = c_hat

        # dict_xy_tau = self.rewrap_xy_tau(xy_tau)
        # tau_hat = self.ecmy.rewrap_ge_dict(dict_xy_tau["ge_x"])["tau_hat"]

        # calculate war values
        wv = self.war_vals(m, theta_dict)

        out = []
        for i in range(self.N):
            ids_j = np.delete(np.arange(self.N), i)
            wv_i = wv[:,i][ids_j]
            out.extend(self.constraint_xy_tau_i(xy_tau, i, wv_i, geq=True))

        if bound=="lower":
            return(np.array(out))
        else:
            return(-1 * np.array(out))

    def est_objective_xy_tau(self, xy_tau):

        dict_xy_tau = self.rewrap_xy_tau(xy_tau)
        ge_x = dict_xy_tau["ge_x"]
        dict_ge_x = self.ecmy.rewrap_ge_dict(ge_x)
        tau_hat = dict_ge_x["tau_hat"]

        out = 0
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    # out += (tau_hat[i, j] - 1) ** 2
                    if tau_hat[i, j] != 1:
                        out += np.abs(np.log(tau_hat[i, j]))  # absolute value of log for cost (note no minus 1 because np.log(1) = 0)
                    else:
                        pass
                    # NOTE: log is nice norm here because it treats .5 and 2 as equidistant from 1 and so on, which is consistent with theory
        # out_sqrt = np.sqrt(out)
        # if out_sqrt > 1000:
        #     print(out_sqrt)
        # out_log = np.log(out)

        if (time.time() - self.clock) > 60:
            print("objective value:")
            print(out)
            print("b:")
            print(dict_xy_tau["b"])
            print("tau_hat:")
            print(tau_hat)
            print("multipliers 0")
            print(self.rewrap_lambda_i(self.rewrap_lambda(dict_xy_tau["lambda_x"])[0]))
            print("minute:" + str(self.minute))
            self.clock = time.time()
            self.minute += 1

        # return(out_sqrt)
        return(out)

    def bounds_xy(self):

        bnds = []
        for i in range(self.N):
            for j in range(self.N):
                bnds.append((1 / self.ecmy.tau[i, j], self.tauMax / self.ecmy.tau[i, j]))  # free trade as lower bound, tauMax as upper bound
        for i in range(self.x_len - self.N**2):
            bnds.append((.01, None))  # positive ge vars
        for i in range(self.lambda_i_len*self.N):  # multipliers
            bnds.append((None, None))
        for i in range(self.N):  # preference parameters
            bnds.append((-1, 2))

        return(bnds)

    def est_xy_tau(self, m, alpha, c_hat, xsv=None):

        self.clock = time.time()

        cons = []
        # cons.append({'type': 'eq', 'fun': self.constraint_xy_tau, 'args':(m, alpha, c_hat, "lower", )})
        # cons.append({'type': 'eq', 'fun': self.constraint_xy_geq, 'args':("lower", )})
        cons.append({'type': 'ineq', 'fun': self.constraint_xy_tau, 'args':(m, alpha, c_hat, "lower", )})
        cons.append({'type': 'ineq', 'fun': self.constraint_xy_tau, 'args':(m, alpha, c_hat, "upper", )})
        # cons.append({'type': 'ineq', 'fun': self.constraint_xy_geq, 'args':("lower", )})
        # cons.append({'type': 'ineq', 'fun': self.constraint_xy_geq, 'args':("upper", )})

        # svs near free trade
        thft = self.tau_hat_ft_i(np.ones_like(self.ecmy.tau), np.arange(self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    thft[i, j] += .1
        ge_dict_thft = self.ecmy.geq_solve(thft, np.ones(self.N))

        if xsv is None:
            dict_xy_tau_sv = dict()
            # dict_xy_tau_sv["ge_x"] = np.random.normal(1, .1, self.x_len)
            dict_xy_tau_sv["ge_x"] = np.ones(self.x_len)
            # dict_xy_tau_sv["ge_x"] = self.ecmy.unwrap_ge_dict(ge_dict_thft) # start at free trade
            # dict_xy_tau_sv["lambda_x"] = np.random.normal(0, .1, self.lambda_i_len*self.N)
            dict_xy_tau_sv["lambda_x"] = np.zeros(self.lambda_i_len*self.N)
            dict_xy_tau_sv["b"] = np.random.normal(.5, .1, self.N)

            xy_tau_sv = self.unwrap_xy_tau(dict_xy_tau_sv)
        else:
            xy_tau_sv = xsv

        mxit = 100000
        bnds = self.bounds_xy()
        out = opt.minimize(self.est_objective_xy_tau, xy_tau_sv, constraints=cons, bounds=bnds, method="SLSQP", options={"maxiter":mxit, "disp": True})

        if out['success'] == True:
            return(out)
        else:
            print("RECURSING...")
            ge_x = self.rewrap_xy_tau(out['x'])["ge_x"]
            print(ge_x)

            # reset large tau values to free trade levels
            tau_hat = self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"]
            print(tau_hat)
            for i in range(self.N):
                for j in range(self.N):
                    if tau_hat[i, j] * self.ecmy.tau[i, j] > self.tauMax:
                        print("resetting...")
                        tau_hat[i, j] = 1 / self.ecmy.tau[i, j]
            ge_dict2 = self.ecmy.geq_solve(tau_hat, np.ones(self.N))
            ge_x2 = self.ecmy.unwrap_ge_dict(ge_dict2)

            noise = np.random.normal(0, .1, len(ge_x2))
            ge_x2 += noise

            # repackage
            dict_xy_tau_sv2 = dict()
            dict_xy_tau_sv2["ge_x"] = ge_x2
            dict_xy_tau_sv2["lambda_x"] = self.rewrap_xy_tau(out['x'])["lambda_x"]
            dict_xy_tau_sv2["b"] = self.rewrap_xy_tau(out['x'])["b"]
            print(dict_xy_tau_sv2)
            print(out)

            return(self.est_xy_tau(m, alpha, c_hat, xsv=self.unwrap_xy_tau(dict_xy_tau_sv2)))

    def est_b_i_obj(self, thi_x_lbda_b, id):

        th_i = thi_x_lbda_b[0:self.N]

        out = 0.0
        for j in range(self.N):
            if id != j:
                # out += (tau_hat[i, j] - 1) ** 2
                if th_i[j] != 1:
                    out += np.log(th_i[j]) ** 2

        if (time.time() - self.clock) > 10:
            print("objective value:")
            print(out)
            print("b:")
            print(thi_x_lbda_b[-1])
            print("lambda_i:")
            print(thi_x_lbda_b[self.x_len-self.N**2:self.x_len-self.N**2+self.lambda_i_len])
            print("thi:")
            print(th_i)
            self.clock = time.time()
            self.minute += 1

        return(out)

    def est_b_i_cons(self, thi_x_lbda_b, id, m, alpha, c_hat, bound="lower"):

        b = thi_x_lbda_b[-1]

        theta_dict = dict()
        theta_dict["b"] = np.repeat(b, self.N)  # NOTE: not right but not worrying about other b now
        theta_dict["alpha"] = alpha
        theta_dict["c_hat"] = c_hat

        # calculate war values
        wv = self.war_vals(m, theta_dict)

        ids_j = np.delete(np.arange(self.N), id)
        wv_i = wv[:,id][ids_j]

        xnt = thi_x_lbda_b[self.N:self.x_len-self.N**2]

        th_i = thi_x_lbda_b[0:self.N]  #id's policies

        lambda_i_x = thi_x_lbda_b[self.x_len-self.N**2:self.x_len-self.N**2+self.lambda_i_len]

        input = []
        input.extend(th_i)
        input.extend(xnt)
        input.extend(lambda_i_x)
        input = np.array(input)

        th = np.ones_like(self.ecmy.tau)
        th[id, ] = th_i

        out = self.Lzeros_tixlbda(input, np.repeat(b, self.N), th, wv_i, id, geq=True)

        if bound == "lower":
            return(1.0*out)
        else:
            return(-1.0*out)

    def est_b_i_bnds(self, id):

        bnds = []
        for j in range(self.N):
            if j != id:
                bnds.append((1 / self.ecmy.tau[id, j], self.tauMax / self.ecmy.tau[id, j]))
            else:
                bnds.append((.99, 1.01))
        for i in range(self.x_len-self.N**2-self.N):
            bnds.append((.1, None))
        for i in range(self.lambda_i_len):
            bnds.append((None, None))

        bnds.append((-1, 2))

        print(bnds)

        return(bnds)

    def est_b_i(self, id, m, alpha, c_hat, sv=None, scipy=True):

        self.clock = time.time()

        if sv is None:

            # start at free trade
            sv = []
            for j in range(self.N):
                if j != id:
                    sv.extend(np.array([1 / self.ecmy.tau[id, j] + .1]))
                else:
                    sv.extend(np.array([1]))

            th = self.tau_hat_ft_i(np.ones_like(self.ecmy.tau), np.array([0]))
            th[id, ] += .1
            th[id, id] = 1
            ge_dict = self.ecmy.geq_solve(th, np.ones(self.N))
            ge_x = self.ecmy.unwrap_ge_dict(ge_dict)

            sv.extend(ge_x[self.N**2+self.N:])
            print(sv)

            # start at ones
            # sv.extend(np.ones(self.N))
            # sv.extend(np.ones(self.x_len-self.N**2-self.N))

            sv.extend(np.zeros(self.lambda_i_len))
            sv.extend(np.array([.9]))
            sv = np.array(sv)

        else:
            pass

        if scipy is True:
            cons = []
            # cons.append({'type': 'eq', 'fun': self.est_b_i_cons, 'args':(id, m, alpha, c_hat, "lower", )})
            cons.append({'type': 'ineq', 'fun': self.est_b_i_cons, 'args':(id, m, alpha, c_hat, "lower", )})
            cons.append({'type': 'ineq', 'fun': self.est_b_i_cons, 'args':(id, m, alpha, c_hat, "upper", )})

            bnds = self.est_b_i_bnds(id)

            mxit = 100000
            out = opt.minimize(self.est_b_i_obj, sv, constraints=cons, bounds=bnds, method="SLSQP", args=(0, ), options={"disp": True, "maxiter":mxit, 'ftol':1e-02})

            if out['success'] == True:
                return(out)
            else:
                print("RECURSING...")
                x = out['x']
                print(out)

                noise = np.random.normal(0, .1, len(x))
                noise[id] = 0  # don't change own policy
                x += noise

                return(self.est_b_i(id, m, alpha, c_hat, sv=x))
        else:
            def obj_wrap(x, grad):
                return(self.est_b_i_obj(x, id))
            def cons_wrap_lower(x, grad):
                return(self.est_b_i_cons(x, id, m, alpha, c_hat, bound="lower"))
            def cons_wrap_upper(x, grad):
                return(self.est_b_i_cons(x, id, m, alpha, c_hat, bound="upper"))

            lb = []
            ub = []
            for j in range(self.N):
                lb.append(1 / self.ecmy.tau[id, j])
                ub.append(self.tauMax / self.ecmy.tau[id, j])
            for i in range(self.x_len - self.N**2 - self.N):
                lb.append(.01)
                ub.append(100)
            for i in range(self.lambda_i_len):
                lb.append(-10000)
                ub.append(10000)
            lb.append(-1)
            ub.append(2)

            o = nlopt.opt(nlopt.LD_SLSQP, len(sv))
            o.set_min_objective(obj_wrap)
            o.add_equality_constraint(cons_wrap_lower)
            # o.add_inequality_constraint(cons_wrap_lower)
            # o.add_inequality_constraint(cons_wrap_upper)
            # o.set_lower_bounds(lb)
            # o.set_upper_bounds(ub)
            print(sv)
            xo = o.optimize(sv)
            return(xo)

    def bounds_y(self):
        """Bounds on parameter, military allocation, multiplier search

        Returns
        -------
        list
            self.y_len list of tuples specifying bounds for estimator

        """

        # TODO bound c_hat between 0 and 1?
        bnds = []
        for i in range(self.y_len):
            if i < self.N:
                bnds.append((0, 2))  # b
            if i >= self.N and i < self.N + self.alpha_len:  # alphas
                bnds.append((None, None))
            # if i == self.N + self.alpha_len:  # gamma
            #     bnds.append((0, 1))
            if i == self.N + self.alpha_len + 1:  # c_hat
                bnds.append((0, 1))
            if i >= self.theta_len and i < self.theta_len + self.N**2: # military allocations bounded below at zero
                bnds.append((0, None))
            if i >= self.theta_len + self.N**2:  # multipliers (need to allow these to take on value of zero, but shouldn't be negative at solution)
                # bnds.append((None, None))
                bnds.append((0, None))

        return(bnds)

    def estimate(self):
        """Parameter estimation problem

        Returns
        -------
        vector
            moment minimizing y vector

        """

        # set timer
        self.clock = time.time()

        # initialize starting values
        theta_dict = dict()
        theta_dict["b"] = np.repeat(1, self.N)
        theta_dict["alpha"] = np.array([0, -.1])
        # theta_dict["gamma"] = np.array([.5])
        theta_dict["c_hat"] = np.array([.25])

        m = np.zeros_like(self.ecmy.tau)
        for i in range(self.N):
            m[i, ] = self.M[i] / self.N  # distribute military allocations equally across all targets

        y_dict = dict()
        y_dict["theta_m"] = self.unwrap_theta(theta_dict)
        y_dict["m"] = self.unwrap_m(m)
        # y_dict["lambda_M"] = np.zeros(self.N)
        # y_dict["lambda_x"] = np.zeros(self.lambda_i_len * self.N)
        y_dict["lambda_M"] = np.random.normal(1, .1, self.N)
        y_dict["lambda_x"] = np.random.normal(1, .1, (self.lambda_i_len * self.N))

        y = self.unwrap_y(y_dict)

        cons = []
        for i in range(self.N):
            cons.append({'type': 'eq', 'fun': self.Lzeros_m_y, 'args':(i, "lower",)})
            # cons.append({'type': 'ineq', 'fun': self.Lzeros_m_y, 'args':(i, "lower",)})
            # cons.append({'type': 'ineq', 'fun': self.Lzeros_m_y, 'args':(i, "upper",)})

        bnds = self.bounds_y()

        mxit = 100000
        out = opt.minimize(self.est_objective, y, constraints=cons, bounds=bnds, method="SLSQP", options={"maxiter":mxit})

        return(out)

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

        m = self.rewrap_m(m_x)

        m_ii = m[i, i]
        m_ji = m[j, i]

        # num_ji = (m_ji * rhoM[j, i]) ** theta_dict["gamma"][0]  # numerator
        # den_ji = (num_ji + m_ii ** theta_dict["gamma"][0])  # denominator
        num_ji = (m_ji * rhoM[j, i])  # numerator
        den_ji = (num_ji + m_ii)  # denominator
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
        rhoM = 1 / (1 + np.exp(-1 * theta_dict["alpha"][0] - self.W * theta_dict["alpha"][1] + epsilon))
        return(rhoM)

    def Lsolve(self, tau_hat, m, theta_dict, id, ft=False, mtd="lm"):
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
        wv = self.war_vals(m, theta_dict)
        wv_i = wv[:,id][j]

        b = theta_dict["b"]

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
                return(self.Lsolve(tau_hat, m, theta_dict, id, ft=ft, mtd="hybr"))
            else:  # otherwise start from free trade
                return(self.Lsolve(tau_hat, m, theta_dict, id, ft=True, mtd="lm"))

    def constraints_tau(self, ge_dict, tau_free, wv_i, b, m=None, ge=True, deficits=False, mil=False):
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

        if m is None:
            m = np.diag(self.M)

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
                    cons.append({'type': 'ineq', 'fun': self.con_mil, 'args':(tau_free, j, m, wv_i[idx], b, )})

        return(cons)

    def con_mil(self, ge_x, i, j, m, wv_ji, b):
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
        G_j = self.G_hat(ge_x, b, ids=np.array([j]), log=False)

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

    def br_war_ji(self, ge_x, b, j, i, mpec=True, full_opt=False):
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

        """

        ge_x = np.copy(ge_x)
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)

        if full_opt == True:
            # initialize starting values of ge_x to equilibrium
            ge_dict = self.ecmy.geq_solve(ge_dict["tau_hat"], ge_dict["D_hat"])
            ge_x = self.ecmy.unwrap_ge_dict(ge_dict)

            mxit = 100000
            ftol = 1e-06
            wv_null = np.repeat(0, self.N - 1)
            cons = self.constraints_tau(ge_dict, i, wv_null, b, mil=False)
            bnds = self.bounds()
            thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, args=(b, np.array([j]), -1, True, ), method="SLSQP", options={"maxiter":mxit, "ftol":ftol})

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
                bnds.append((tauHatMin[i,j], tauHatMax[i,j]))
        for i in range(self.ecmy.ge_x_len-self.N**2):
            bnds.append((None, None))

        return(bnds)


    def br(self, ge_x, b, m, wv_i, id, mil=True, method="SLSQP"):
        """Calculate optimal policies for gov id, given others' policies in ge_x.

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function for order of values.
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

        # initialize starting values of ge_x to equilibrium
        ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
        ge_dict = self.ecmy.geq_solve(ge_dict["tau_hat"], ge_dict["D_hat"])
        ge_x = self.ecmy.unwrap_ge_dict(ge_dict)
        mxit = 100000
        # if mpec == True:

        cons = self.constraints_tau(ge_dict, id, wv_i, b, m=m, mil=mil)
        thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, args=(b, np.array([id]), -1, True, ), method="SLSQP", options={"maxiter":mxit})
        # else:
        #     cons = self.constraints_tau(ge_dict, id, ge=False, mil=True)
        #     bnds = self.bounds()
        #     thistar = opt.minimize(self.G_hat, ge_x, constraints=cons, bounds=bnds, args=(self.b, np.array([id]), -1, False, ), method="SLSQP", options={"maxiter":mxit})

        return(thistar['x'])

    def loss_tau_i(self, tau_i):
        """Loss function for b estimation, absolute log loss

        Parameters
        ----------
        tau_i : vector length N
            i's policies

        Returns
        -------
        float
            loss

        """

        out = np.sum(np.abs(np.log(tau_i)))

        return(out)

    def est_b_i_grid(self, id, b_init, m, theta_dict, epsilon):

        bmax = np.max(self.b_vals)
        bmin = np.min(self.b_vals)
        b_vec = np.copy(b_init)

        b = b_vec[id]

        stop = False
        while stop is False:

            # first
            idx_first = helpers.which_nearest(self.b_vals, b)
            idx_up = idx_first + 1
            idx_down = idx_first - 1

            b = self.b_vals[idx_first]

            print("bmax: " + str(bmax))
            print("bmin: " + str(bmin))

            Loss = []
            for idx in [idx_down, idx_first, idx_up]:
                b_idx = self.b_vals[idx]
                print(b_idx)
                b_vec[id] = b_idx
                wv = self.war_vals(b_vec, m, theta_dict, epsilon)
                ids_j = np.delete(np.arange(self.N), id)
                wv_i = wv[:,id][ids_j]
                br = self.br(np.ones(self.x_len), b_vec, m, wv_i, id)
                br_dict = self.ecmy.rewrap_ge_dict(br)
                tau_i = br_dict["tau_hat"][id, ]
                print(tau_i)
                loss = self.loss_tau_i(tau_i)
                Loss.append(loss)

            print(Loss)
            if Loss[1] < Loss[2] and Loss[1] < Loss[0]:
                stop = True
            else:
                if Loss[2] < Loss[1]:
                    bmin = b
                    b = (bmax - b) / 2 + bmin
                if Loss[0] < Loss[1]:
                    bmax = b
                    b = (b - bmin) / 2 + bmin
            print("b: " + str(b))
            print("bmin: " + str(bmin))
            print("bmax: " + str(bmax))

        return(b)


    def br_cor(self, ge_x, m, mpec=True):
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


        tau_hat_br = np.ones_like(self.ecmy.tau)
        D_hat = np.ones(self.N)

        for i in range(self.N):
            ge_x = self.br(ge_x, m, i, mpec=mpec)
            ge_dict = self.ecmy.rewrap_ge_dict(ge_x)
            tau_hat_br[i, ] = ge_dict["tau_hat"][i, ]

        ge_dict = self.ecmy.geq_solve(tau_hat_br, D_hat)
        ge_x = self.ecmy.unwrap_ge_dict(ge_dict)

        return(ge_x)

    def nash_eq(self, m):
        """Calculates Nash equilibrium of policy game

        Returns
        -------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs at NE values.

        """

        ge_x_sv = np.ones(self.ecmy.ge_x_len)
        ge_x_out = opt.fixed_point(self.br_cor, ge_x_sv, args=(m, True, ), method="del2")
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

    def pop_rc_vals(self):
        """Generate regime change value matrix for each value of preference parameters in self.b_vals

        Returns
        -------
        dict
            Dictionary mapping b vals to to matrices of war values. Each entry is war value for row in war against column

        """

        wv = dict()
        ge_x = np.ones(self.ecmy.ge_x_len)
        for b in self.b_vals:
            print(b)
            b_vec = np.repeat(b, self.N)
            wvb = np.zeros_like(self.ecmy.tau)
            for i in range(self.N):
                for j in range(self.N):
                    if i != j:
                        print(str(i) + " " + str(j))
                        # start_time = time.time()
                        # populates matrix column-wise, value for row of controlling policy in column
                        ge_br_war_ji = self.br_war_ji(ge_x, b_vec, j, i, full_opt=True)
                        G_hat_ji = self.G_hat(ge_br_war_ji, b_vec, ids=np.array([j]))
                        wvb[j, i] = G_hat_ji
                        # print(time.time() - start_time)
            wv[b] = wvb
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
                b = self.b_vals[tick]
                vals = [float(i) for i in row]
                wv[b] = np.reshape(np.array(vals), (self.N, self.N))
                tick += 1
        return(wv)
