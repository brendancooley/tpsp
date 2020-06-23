import numpy as np
import imp

import c_results as results
import c_policies as policies
imp.reload(results)

location = "local"
size = "mid/"

results_0 = results.results(location, size)
pecmy_0 = policies.policies(results_0.data, results_0.params, results_0.ROWname, 0)

v_500 = np.genfromtxt(results_0.setup.quantiles_v_path, delimiter=",")[1]
gamma_500 = np.genfromtxt(results_0.setup.quantiles_gamma_path, delimiter=",")[1]
alpha1_500 = np.genfromtxt(results_0.setup.quantiles_alpha1_path, delimiter=",")[1]
alpha2_500 = np.genfromtxt(results_0.setup.quantiles_alpha2_path, delimiter=",")[1]

theta_dict = dict()
theta_dict["eta"] = results_0.setup.eta
theta_dict["c_hat"] = results_0.setup.c_hat
theta_dict["alpha1"] = alpha1_500
theta_dict["alpha2"] = alpha2_500
theta_dict["gamma"] = gamma_500
theta_dict["C"] = np.repeat(results_0.setup.c_hat, results_0.N)

theta_x = pecmy_0.unwrap_theta(theta_dict)

### COUNTERFACTUAL 1: DEMILITARIZATION ###

# results_1 = results.results(location, size)
# pecmy_1 = policies.policies(results_1.data, results_1.params, results_1.ROWname, 0)
#
# xlhvt_prime_1 = results_1.compute_counterfactual(v_500, theta_x, pecmy_1.mzeros)
# np.savetxt(results_1.setup.cfct_demilitarization_path + "x.csv", xlhvt_prime_1, delimiter=",")
#
# ge_x_star_1 = pecmy_1.rewrap_xlhvt(xlhvt_prime_1)["ge_x"]
# X_star_1 = pecmy_1.ecmy.rewrap_ge_dict(ge_x_star_1)["X_hat"] * pecmy_1.ecmy.Xcif  # counterfactual trade flows
# np.savetxt(results_1.setup.cfct_demilitarization_path + "X_prime.csv", X_star_1, delimiter=",")

### COUNTERFACTUAL 2: CHINESE MILITARY EXPANSION ###

results_2 = results.results(location, size)

CHN_id = np.where(results_2.data["ccodes"]=="CHN")
USA_id = np.where(results_2.data["ccodes"]=="USA")
results_2.data["M"][CHN_id] = results_2.data["M"][USA_id]

pecmy_2 = policies.policies(results_2.data, results_2.params, results_2.ROWname, 0, tau_buffer=.75)

xlhvt_prime_2 = results_2.compute_counterfactual(v_500, theta_x, pecmy_2.m, tau_bounds=False, ge_ones=True)
np.savetxt(results_1.setup.cfct_china_path + "x.csv", xlhvt_prime_2, delimiter=",")

ge_x_star_2 = pecmy_2.rewrap_xlhvt(xlhvt_prime_2)["ge_x"]
X_star_2 = pecmy_2.ecmy.rewrap_ge_dict(ge_x_star_2)["X_hat"] * pecmy_2.ecmy.Xcif  # counterfactual trade flows
np.savetxt(results_2.setup.cfct_china_path + "X_prime.csv", X_star_2, delimiter=",")
