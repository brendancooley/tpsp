import sys
import os

helpersPath = os.path.expanduser("../source/")
sys.path.insert(1, helpersPath)

import helpers

class setup:

    def __init__(self, location, size, bootstrap=False, bootstrap_id=0, mil_off=False, base_path=None):

        # CALIBRATED PARAMETERS

        self.eta = 1.5
        self.c_hat = 25

        # MACRO PATHS

        if base_path is None:
            mkdirs = True
            base_path = os.path.expanduser('~')
        else:
            mkdirs = False

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

        self.figs_path_0 = "../02_figs/"

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
        self.M2030_path = self.results_path_base + "M2030.csv"

        self.quantiles_v_path = self.estimates_path_base + "quantiles_v.csv"
        self.quantiles_gamma_path = self.estimates_path_base + "quantiles_gamma.csv"
        self.quantiles_alpha1_path = self.estimates_path_base + "quantiles_alpha1.csv"
        self.quantiles_alpha2_path = self.estimates_path_base + "quantiles_alpha2.csv"
        self.quantiles_rcv_path = self.estimates_path_base + "quantiles_rcv.csv"
        self.quantiles_peace_probs_path = self.estimates_path_base + "quantiles_peace_probs.csv"
        self.quantiles_tau_path = self.estimates_path_base + "quantiles_tau.csv"
        self.quantiles_Ghat_path = self.estimates_path_base + "quantiles_Ghat.csv"
        self.quantiles_Uhat1_path = self.estimates_path_base + "quantiles_Uhat1.csv"

        # COUNTERFACTUAL PATHS

        self.cfct_demilitarization_path = self.counterfactuals_path + "demilitarization/"
        self.cfct_china_path = self.counterfactuals_path + "china/"
        self.cfct_china_v_path = self.counterfactuals_path + "china_v/"
        self.cfct_us_path = self.counterfactuals_path + "us/"

        # FIGURE PATHS

        self.f_ccodes_path = self.figs_path_0 + "ccodes.png"
        self.f_cfact_demilitarization_Xprime_path = self.figs_path_0 + "cfact_demilitarization_Xprime.png"
        self.f_cfact_demilitarization_G_path = self.figs_path_0 + "cfact_demilitarization_G.png"
        self.f_cfact_demilitarization_U_path = self.figs_path_0 + "cfact_demilitarization_U.png"
        self.f_cfact_china_Xprime_path = self.figs_path_0 + "cfact_china_Xprime.png"
        self.f_cfact_china_G_path = self.figs_path_0 + "cfact_china_G.png"
        self.f_cfact_china_U_path = self.figs_path_0 + "cfact_china_U.png"
        self.f_cfact_china_tau_path = self.figs_path_0 + "cfact_china_tau.png"
        self.f_estimates_pref_path = self.figs_path_0 + "estimates_pref.png"
        self.f_estimates_pref_mo_path = self.figs_path_0 + "estimates_pref_mo.png"
        self.f_estimates_mil_path = self.figs_path_0 + "estimates_mil.png"
        self.f_fit_path = self.figs_path_0 + "fit.png"
        self.f_fit_eps_path = self.figs_path_0 + "fit_eps.png"
        self.f_milex_path = self.figs_path_0 + "milex.png"
        self.f_pr_peace_path = self.figs_path_0 + "pr_peace.png"
        self.f_rcv_path = self.figs_path_0 + "rcv.png"
        self.f_tau_epbt_path = self.figs_path_0 + "tau_epbt.png"
        self.f_tau_rf_dw_path = self.figs_path_0 + "tau_rf_dw.png"
        self.f_tau_rf_table_path = self.figs_path_0 + "tau_rf_table.png"

        # MAKE DIRECTORIES

        if mkdirs == True:
            helpers.mkdir(self.results_path_base)
            helpers.mkdir(self.estimates_path)
            helpers.mkdir(self.counterfactuals_path)
            helpers.mkdir(self.cfct_demilitarization_path)
            helpers.mkdir(self.cfct_china_path)
            helpers.mkdir(self.cfct_us_path)
            helpers.mkdir(self.cfct_china_v_path)
