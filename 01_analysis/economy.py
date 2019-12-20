import autograd.numpy as np
from scipy import optimize as opt
import imp
import autograd as ag

import helpers as hp

class economy:

    def __init__(self, data, params):

        # Data
        self.tau = data["tau"]
        self.D = data["D"]

        self.Xcif = data["Xcif"]
        self.Y = data["Y"]
        self.E = data["E"]
        self.r = data["r"]
        self.lambda_pc = self.lambda_pc_f()

        # Params
        self.beta = params["beta"]
        self.theta = params["theta"]
        self.mu = params["mu"]
        self.nu = params["nu"]

        # Other
        self.N = len(self.Y)
        self.x0_sv = np.ones(self.N**2+4*self.N)
        self.ge_x_len = 2*self.N**2+5*self.N

    def unwrap_ge_dict(self, ge_dict):
        """Convert dictionary storing GE inputs and outputs into flattened vector. Use rewrap_ge_dict to re-convert flattened vector to dictionary.

        Parameters
        ----------
        ge_dict : dictionary
            Dictionary storing GE inputs and outputs. See function for keys.

        Returns
        -------
        vector
            1d numpy array storing flattened ge inputs and outputs. See function for order of values.

        """

        x = []
        x.extend(ge_dict["tau_hat"].ravel())
        x.extend(ge_dict["D_hat"])
        x.extend(ge_dict["X_hat"].ravel())
        x.extend(ge_dict["P_hat"])
        x.extend(ge_dict["w_hat"])
        x.extend(ge_dict["r_hat"])
        x.extend(ge_dict["E_hat"])

        return(np.array(x))

    def rewrap_ge_dict(self, ge_x):
        """Convert flattened vector storing GE inputs and outputs into dictionary. Use unwrap_ge_dict to convert back to flattened vector

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function for order of values.

        Returns
        -------
        dictionary
            Dictionary storing GE inputs and outputs. See function for keys.

        """

        ge_dict = dict()
        ge_dict["tau_hat"] = np.reshape(ge_x[0:self.N**2], (self.N, self.N))
        ge_dict["D_hat"] = ge_x[self.N**2:self.N**2+self.N]
        ge_dict["X_hat"] = np.reshape(ge_x[self.N**2+self.N:2*self.N**2+self.N], (self.N, self.N))
        ge_dict["P_hat"] = ge_x[2*self.N**2+self.N:2*self.N**2+2*self.N]
        ge_dict["w_hat"] = ge_x[2*self.N**2+2*self.N:2*self.N**2+3*self.N]
        ge_dict["r_hat"] = ge_x[2*self.N**2+3*self.N:2*self.N**2+4*self.N]
        ge_dict["E_hat"] = ge_x[2*self.N**2+4*self.N:2*self.N**2+5*self.N]

        return(ge_dict)

    def update_ge_dict(self, x0, ge_dict):
        """Short summary.

        Parameters
        ----------
        x0 : vector
            Endogenous values of trade flows, prices, wages, revenues, expenditures. Flattened vector.
        ge_dict : dictionary
            Dictionary storing GE inputs and outputs. See function for keys.

        Returns
        -------
        ge_dict : dictionary
            Updated dictionary storing GE inputs and outputs. Output values replaced with values from x0.

        """

        ge_dict["X_hat"] = np.reshape(x0[0:self.N**2], (self.N, self.N))
        ge_dict["P_hat"] = x0[self.N**2:self.N**2+self.N]
        ge_dict["w_hat"] = x0[self.N**2+self.N:self.N**2+2*self.N]
        ge_dict["r_hat"] = x0[self.N**2+2*self.N:self.N**2+3*self.N]
        ge_dict["E_hat"] = x0[self.N**2+3*self.N:self.N**2+4*self.N]

        return(ge_dict)

    def geq_f(self, x0, ge_dict):
        """Returns between equilibrium values and input values. Fixed point of this function is GE solution

        Parameters
        ----------
        x0 : vector
            Endogenous components of ge_dict, length N^2 + 4*N.
        ge_dict : dictionary
            Dictionary storing GE inputs and outputs. See function for keys.

        Returns
        -------
        vector :
            Length N^2 + 4*N. Difference between starting values and endogenous output

        """

        ge_dict = self.update_ge_dict(x0, ge_dict)
        ge_x = self.unwrap_ge_dict(ge_dict)

        out = self.geq_diffs(ge_x)

        return(out)

    def geq_diffs(self, ge_x, bound="lower"):
        """Short summary.

        Parameters
        ----------
        ge_x : vector
            1d numpy array storing flattened ge inputs and outputs. See function for order of values.
        bound : "lower" or "upper"
            Order of differences.

        Returns
        -------
        vector
            Differences between endogenous inputs and equilibrium values

        """

        ge_dict = self.rewrap_ge_dict(ge_x)
        out = []

        Xdiff = self.X_hat(ge_dict) - ge_dict["X_hat"]
        out.extend(Xdiff.ravel())

        Pdiff = self.P_hat(ge_dict) - ge_dict["P_hat"]
        out.extend(Pdiff)

        wdiff = self.w_hat(ge_dict) - ge_dict["w_hat"]
        out.extend(wdiff)

        rdiff = self.r_hat(ge_dict) - ge_dict["r_hat"]
        out.extend(rdiff)

        Ediff = self.E_hat(ge_dict) - ge_dict["E_hat"]
        out.extend(Ediff)

        out = np.array(out)
        if bound != "lower":  # flip signs if we're looking for upper bound
            out = out*-1

        return(out)

    def geq_solve(self, tau_hat, D_hat, fct=1, mtd="hybr"):
        """Short summary.

        Parameters
        ----------
        tau_hat : matrix
            N times N numpy array of trade policy changes
        D_hat : vector
            Length N numpy array of deficit changes
        fct : scalar
            \in [.1, 100] controls step size of root solver. Function recursively reduces this if no solution found.

        Returns
        -------
        ge_dict : dictionary
            Equilibrium dictionary storing GE inputs and outputs.

        """

        ge_dict = dict()
        ge_dict["tau_hat"] = tau_hat
        ge_dict["D_hat"] = D_hat

        # geq_sol = opt.fsolve(self.geq_f, x0=self.x0_sv, full_output=True, args=(ge_dict.copy(),), factor=fct)
        geq_sol = opt.root(self.geq_f, x0=self.x0_sv, args=(ge_dict.copy(),), method=mtd, options={"factor":fct})  # opt.root version, hybr seems to work faster than lm, but lm can solve problems hybr can't

        # x_out = geq_sol[0]
        x_out = geq_sol['x']  # opt.root version

        # if geq_sol[2] == 1: # return solution if found
        if geq_sol["success"] == True:  # opt.root version
            ge_dict = self.update_ge_dict(x_out, ge_dict)
            ge_dict["r_hat"][(-.0001 < ge_dict["r_hat"]) & (ge_dict["r_hat"] < .0001)] = 0  # replace small revenue counterfactuals with zeros
            return(ge_dict)
        else:
            if fct / 2 > .1: # recurse with smaller steps
                return(self.geq_solve(tau_hat, D_hat, fct=fct/2, mtd=mtd))
            else:
                if mtd == "lm":
                    return("Solution not found.", geq_sol) # return 0
                else:
                    return(self.geq_solve(tau_hat, D_hat, mtd="lm"))

    def U_hat(self, ge_dict):
        """Short summary.

        Parameters
        ----------
        ge_dict : dictionary
            Equilibrium dictionary storing GE inputs and outputs.

        Returns
        -------
        U_hat : vector
            Length N vector of welfare changes

        """

        eu_hat = self.Eu_hat(ge_dict)
        Pcd_hat = self.Pcd_hat(ge_dict)
        U_hat = eu_hat / Pcd_hat

        return(U_hat)

    def update_ecmy(self, tau_hat, D_hat):

        ge_dict_out = self.geq_solve(tau_hat, D_hat)

        self.tau = self.tau * ge_dict_out["tau_hat"]
        self.D = self.D * D_hat

        self.Xcif = self.Xcif * ge_dict_out["X_hat"]
        self.Y = self.Y * ge_dict_out["w_hat"]
        self.E = self.E * ge_dict_out["E_hat"]
        self.r = self.r * ge_dict_out["r_hat"]
        self.lambda_pc = self.lambda_pc_f()

    def purgeD(self):
        """Solves model for deficit free world and replaces economy primitives with counterfactual values.

        """

        D_hat_zeros = np.zeros(self.N)
        tau_hat_ones = np.ones_like(self.tau)

        self.update_ecmy(tau_hat_ones, D_hat_zeros)



    def lambda_pc_f(self):
        """calculates trade shares (post-customs)

        Returns
        -------
        vector
            N times 1 vector of expenditure shares, evaluated post customs

        """

        tauX = self.tau * self.Xcif
        Einv = np.diag(np.power(self.E, -1))

        lambda_pcT = np.dot(tauX.T, Einv)
        lambda_pc = lambda_pcT.T

        return(lambda_pc)


    def X_hat(self, ge_dict):
        """

        Parameters
        ----------
        ge_dict : dictionary
            Dictionary storing GE inputs and outputs. See unwrap_ge_dict for keys.

        Returns
        -------
        matrix
            N times N matrix of changes in trade flows.

        """

        w_hat, P_hat, E_hat, tau_hat = ge_dict["w_hat"], ge_dict["P_hat"], ge_dict["E_hat"], ge_dict["tau_hat"]

        if np.any(ge_dict["P_hat"] < 0):  # nudge to avoid negative trade solutions
            ge_dict["P_hat"][ge_dict["P_hat"] < 0] = .01
        if np.any(ge_dict["w_hat"] < 0):
            ge_dict["w_hat"][ge_dict["w_hat"] < 0] = .01

        A = np.power(tau_hat, -self.theta-1)
        B = np.dot(np.diag(np.power(w_hat, 1-self.beta-self.theta)), np.diag(np.power(P_hat, self.beta-self.theta)))
        C = np.dot(np.diag(np.power(P_hat, self.theta)), np.diag(E_hat))

        AB = np.dot(A, B)
        XhatT = np.dot(AB.T, C)
        Xhat = XhatT.T

        return(Xhat)

    def P_hat(self, ge_dict):
        """

        Parameters
        ----------
        ge_dict : dictionary
            Dictionary storing GE inputs and outputs. See unwrap_ge_dict for keys.

        Returns
        -------
        vector
            N times 1 vector of price changes

        """

        w_hat, P_hat, tau_hat = ge_dict["w_hat"], ge_dict["P_hat"], ge_dict["tau_hat"]

        if np.any(ge_dict["P_hat"] < 0):  # nudge to avoid negative trade solutions
            ge_dict["P_hat"][ge_dict["P_hat"] < 0] = .01

        A = self.lambda_pc * np.power(tau_hat,-self.theta)
        b = np.power(w_hat, 1-self.beta-self.theta) * np.power(P_hat, self.beta-self.theta)

        Phat_int = np.dot(A, b)
        Phat = np.power(Phat_int, -1/self.theta)

        return(Phat)

    def w_hat(self, ge_dict):
        """

        Parameters
        ----------
        ge_dict : dictionary
            Dictionary storing GE inputs and outputs. See unwrap_ge_dict for keys.

        Returns
        -------
        vector
            N times 1 vector of wage changes

        """

        X_hat, r_hat, tau_hat = ge_dict["X_hat"], ge_dict["r_hat"], ge_dict["tau_hat"]

        XcifPrime = X_hat * self.Xcif
        XcifmuPrime = self.tau * tau_hat * XcifPrime - self.mu * XcifPrime * (self.tau * tau_hat - 1)

        a = np.sum(XcifmuPrime, axis=0) * (1 - self.beta)
        b = (1 - self.nu) * self.r * r_hat
        c = 1 / (self.nu * self.Y)

        what = c * (a + b)

        # normalization
        wgdp = np.sum(self.Y)
        y_weights = self.Y / np.sum(self.Y)
        wavg = np.sum(what*y_weights)
        whatn = 1 / wavg * what

        return(whatn)

    def r_hat(self, ge_dict):
        """

        Parameters
        ----------
        ge_dict : dictionary
            Dictionary storing GE inputs and outputs. See unwrap_ge_dict for keys.

        Returns
        -------
        vector
            N times 1 vector of revenue changes

        """

        E_hat, X_hat, tau_hat = ge_dict["E_hat"], ge_dict["X_hat"], ge_dict["tau_hat"]

        if np.sum(self.r) != 0:
            a = np.sum(X_hat * self.Xcif, axis=1)  # axis = 1, sum over imports
            b = np.sum(self.tau * self.Xcif * tau_hat * X_hat, axis=1)
            c = self.mu / self.r

            rhat = c * (b - a)

        else:
            rhat = np.ones_like(self.r)

        return(rhat)

    def E_hat(self, ge_dict):
        """

        Parameters
        ----------
        ge_dict : dictionary
            Dictionary storing GE inputs and outputs. See unwrap_ge_dict for keys.

        Returns
        -------
        vector
            N times 1 vector of changes in tradable expenditure.

        """

        X_hat, r_hat, w_hat, tau_hat, D_hat = ge_dict["X_hat"], ge_dict["r_hat"], ge_dict["w_hat"], ge_dict["tau_hat"], ge_dict["D_hat"]

        XcifPrime = X_hat * self.Xcif
        XcifmuPrime = self.tau * tau_hat * XcifPrime - self.mu * XcifPrime * (self.tau * tau_hat - 1)

        Eq_prime = self.nu * (self.Y * w_hat + self.r * r_hat)  # consumer income spent on tradables
        Ex_prime = self.beta * np.sum(XcifmuPrime, axis=0)  # intermediates share in sum over exports

        E_prime = Eq_prime + Ex_prime + self.D * D_hat

        Ehat = E_prime / self.E

        return(Ehat)

    def Eu_hat(self, ge_dict):
        """

        Parameters
        ----------
        ge_dict : dictionary
            Dictionary storing GE inputs and outputs. See unwrap_ge_dict for keys.

        Returns
        -------
        vector
            N times 1 vector of total consumer expenditure changes.

        """

        w_hat, r_hat, D_hat = ge_dict["w_hat"], ge_dict["r_hat"], ge_dict["D_hat"]

        Eu = self.Y + self.r + self.D  # total consumer expenditure

        a = (self.Y * w_hat + self.r * r_hat)
        b = self.D * D_hat
        c = 1 / Eu

        euhat = c * (a + b)

        return(euhat)

    def Pcd_hat(self, ge_dict):
        """

        Parameters
        ----------
        ge_dict : dictionary
            Dictionary storing GE inputs and outputs. See unwrap_ge_dict for keys.

        Returns
        -------
        vector
            N times 1 vector of CD price index changes.

        """

        P_hat, w_hat = ge_dict["P_hat"], ge_dict["w_hat"]
        if np.any(P_hat == 0):
            print(P_hat)
        if np.any(w_hat == 0):
            print(w_hat)

        pcdhat = np.power(P_hat, self.nu) * np.power(w_hat, 1 - self.nu)

        return(pcdhat)