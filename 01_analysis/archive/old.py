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

    def Lzeros_theta(self, theta_lbda_chi):
        """Short summary.

        Parameters
        ----------
        theta_lbda_m : vector
            v, c_hat, alpha, gamma, lambda_chi

        Returns
        -------
        type
            Description of returned object.

        """

        v = theta_lbda_chi[0:self.N]
        theta = theta_lbda_chi[self.N:self.N+3]
        lambda_chi = np.reshape(theta_lbda_chi[self.N+3:], (self.N, self.N))

        theta_dict = dict()
        theta_dict["c_hat"] = theta[0]
        theta_dict["alpha"] = theta[1]
        theta_dict["gamma"] = theta[2]

        m = self.M / np.ones((self.N, self.N))
        m = m.T
        # NOTE: setting ROW to zero seems to mess with autograd
        # m[self.ROW_id,:] = 0
        # m[:,self.ROW_id] = 0
        # m[self.ROW_id,self.ROW_id] = 1

        # print(theta_dict)
        wv = self.war_vals(v, m, theta_dict, np.zeros((self.N, self.N)))
        # print(wv)

        loss = 0
        for i in range(self.N):
            lambda_dict_i = self.rewrap_lambda_i(np.zeros(self.lambda_i_len))
            lambda_dict_i["chi_i"] = lambda_chi[i, ]
            lambda_x_i = self.unwrap_lambda_i(lambda_dict_i)
            ge_x_lbda_i_x = np.concatenate((np.ones(self.x_len), lambda_x_i))
            Lzeros_i = self.Lzeros(ge_x_lbda_i_x, v, np.ones((self.N, self.N)), wv[:,i], i)
            loss += np.sum(Lzeros_i ** 2)

        self.tick += 1
        if self.tick == 25:
            print("lambda_chi:")
            print(lambda_chi)
            print("v:")
            print(v)
            print("theta:")
            print(theta)
            print("loss:")
            print(loss)
            self.tick = 0

        return(loss)

    def Lzeros_theta_grad(self, x):
        Lzeros_theta_grad_f = ag.grad(self.Lzeros_theta)
        return(Lzeros_theta_grad_f(x))

    def Lzeros_theta_min(self, theta_dict_init, v_init):

        # theta_lbda_chi_init = np.zeros(self.N+3+self.N**2)
        theta_lbda_chi_init = np.ones(self.N+3+self.N**2)
        theta_lbda_chi_init[0:self.N] = v_init
        theta_lbda_chi_init[self.N] = theta_dict_init["c_hat"]
        theta_lbda_chi_init[self.N+1] = theta_dict_init["alpha"]
        theta_lbda_chi_init[self.N+2] = theta_dict_init["gamma"]

        bounds = []
        for i in range(self.N):
            bounds.append((1, None))
        for i in range(3):
            bounds.append((0, None))
        for i in range(self.N**2):
            bounds.append((None, None))


        out = opt.minimize(self.Lzeros_theta, theta_lbda_chi_init, method="TNC", jac=self.Lzeros_theta_grad, bounds=bounds)
        # out = opt.minimize(self.Lzeros_theta, theta_lbda_chi_init, method="TNC", bounds=bounds)

        return(out)
