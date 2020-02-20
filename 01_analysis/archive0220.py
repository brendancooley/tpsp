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
