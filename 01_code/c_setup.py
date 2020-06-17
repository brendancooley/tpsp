import sys
import os

helpersPath = os.path.expanduser("../source/")
sys.path.insert(1, helpersPath)

import helpers

class setup:

    def __init__(self, location, size, bootstrap=False, bootstrap_id=0, mil_off=False):

        # MACRO PATHS

        base_path = os.path.expanduser('~')

        if location == "local":
            self.project_files = base_path + "/Dropbox (Princeton)/1_Papers/tpsp/01_files/"
        if location == "hpc":
            self.project_files = base_path + "/tpsp/"

        self.data_path_0 = self.project_files + "data/"
        self.results_path_0 = self.project_files + "results/"

        self.data_path_base = self.data_path_0 + size
        self.results_path_base = self.results_path_0 + size

        if mil_off == False:
            self.estimates_path_base = self.results_path_base + "estimates/"
        else:
            self.estimates_path_base = self.results_path_base + "estimates_mil_off/"

        if bootstrap == True:
            self.data_path = self.data_path_base + str(bootstrap_id) + "/"
            self.estimates_path = self.estimates_path_base + str(bootstrap_id) + "/"
        else:
            self.data_path = self.data_path_base
            self.estimates_path = self.estimates_path_base

        self.counterfactuals_path = self.results_path_base + "counterfactuals/"
        self.xlhvt_star_path = self.estimates_path + "x.csv"

        # DATA (STATIC)

        self.icews_reduced_path = "~/Dropbox (Princeton)/Public/reducedICEWS/"
        self.icews_counts_path = self.data_path_0 + "icews_counts.csv"

        # DATA (DYNAMIC)

        self.ccodes_path = self.data_path + "ccodes.csv"
        self.ROWname_path = self.data_path + "ROWname.csv"

        self.beta_path = self.data_path + "beta.csv"
        self.theta_path = self.data_path + "theta.csv"
        self.mu_path = self.data_path + "mu.csv"
        self.nu_path = self.data_path + "nu.csv"

        self.tau_path = self.data_path + "tau.csv"
        self.Xcif_path = self.data_path + "Xcif.csv"
        self.Y_path = self.data_path + "y.csv"
        self.Eq_path = self.data_path + "Eq.csv"
        self.Ex_path = self.data_path + "Ex.csv"
        self.r_path = self.data_path + "r.csv"
        self.D_path = self.data_path + "D.csv"

        self.dists_path = self.data_path + "cDists.csv"
        self.M_path = self.data_path + "milex.csv"

        # MAKE DIRECTORIES

        helpers.mkdir(self.results_path_base)
        helpers.mkdir(self.estimates_path)
        helpers.mkdir(self.counterfactuals_path)
