import sys
import os

helpersPath = os.path.expanduser("../source/")
sys.path.insert(1, helpersPath)

import helpers

class setup:

    def __init__(self, location, size, bootstrap=False, bootstrap_id=0, mil_off=False, base_path=None):

        # MACRO PATHS

        if base_path is None:
            mkdirs = False
            base_path = os.path.expanduser('~')
        else:
            mkdirs = True

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
        self.gdp_raw_path = self.data_path_base + "gdp_raw.csv"

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
        self.D_path = self.data_path + "d.csv"

        self.dists_path = self.data_path + "cDists.csv"
        self.M_path = self.data_path + "milex.csv"

        # OUTPUT

        self.icews_counts_path = self.data_path_0 + "icews_counts.csv"
        self.rcv_ft_path = self.results_path_base + "rcv_ft.csv"

        self.quantiles_v_path = self.estimates_path_base + "quantiles_v.csv"
        self.quantiles_gamma_path = self.estimates_path_base + "quantiles_gamma.csv"
        self.quantiles_alpha1_path = self.estimates_path_base + "quantiles_alpha1.csv"
        self.quantiles_alpha2_path = self.estimates_path_base + "quantiles_alpha2.csv"
        self.quantiles_rcv_path = self.estimates_path_base + "quantiles_rcv.csv"
        self.quantiles_peace_probs_path = self.estimates_path_base + "quantiles_peace_probs.csv"

        # MAKE DIRECTORIES

        if mkdirs == True:
            helpers.mkdir(self.results_path_base)
            helpers.mkdir(self.estimates_path)
            helpers.mkdir(self.counterfactuals_path)
