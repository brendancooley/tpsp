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

    def apply_new(_X):
        return(True)

    # Sparse Jacobian
    # NOTE: starting values sometimes induce sparsity for elements that have positive derivatives for some parameters. But problem seems to go away if we make wv_min low enough
        # attempting both versions of sparsity on mini problem
        # doit results: Sparse (TODO: need to debug...runs forever for some reason)
        # python: full
        # (holding gamma fixed at 1)
    # g_sparsity_bin = self.g_sparsity_bin(xlvt_sv)
    # g_sparsity_indices_a = self.g_sparsity_idx(g_sparsity_bin)
    # g_sparsity_indices = (g_sparsity_indices_a[:,0], g_sparsity_indices_a[:,1])


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

    def comp_slack_lbda(self, ge_x_lbda_i_x, v, wv, id):

        ge_x = ge_x_lbda_i_x[0:self.x_len]
        lambda_i_x = ge_x_lbda_i_x[self.x_len:]

        war_diffs = self.war_diffs(ge_x, v, wv, id)
        comp_slack = war_diffs * self.rewrap_lbda_i(lambda_i_x)["chi_i"]

        return(comp_slack)

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

    def geq_diffs_lbda(self, xlsh):
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

        xlsh_dict = self.rewrap_lbda_i_x(xlsh)
        ge_x = xlsh_dict["ge_x"]

        out = self.ecmy.geq_diffs(ge_x)

        return(out)

    def war_diffs_lbda(self, xlsh, id, m, v, theta_dict):
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

        xlsh_dict = self.rewrap_lbda_i_x(xlsh)

        ge_x = xlsh_dict["ge_x"]
        s_i = xlsh_dict["s_i"]
        h = xlsh_dict["h"]

        tau_hat_tilde = self.ecmy.rewrap_ge_dict(ge_x)["tau_hat"]
        wv = self.wv_xlsh(tau_hat_tilde, h, id, m, v, theta_dict)

        wd = self.war_diffs(ge_x, v, wv, id) - s_i

        return(wd)

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

        ge_x_lbda_i_dict = self.rewrap_lbda_i_x(ge_x_lbda_i_x)

        ge_x = ge_x_lbda_i_dict["ge_x"]
        lambda_i_x = ge_x_lbda_i_dict["lambda_i"]
        s_i = ge_x_lbda_i_dict["s_i"]

        war_diffs = self.war_diffs(ge_x, v, wv, id)
        comp_slack = s_i * self.rewrap_lbda_i(lambda_i_x)["chi_i"]

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

        return(out)
