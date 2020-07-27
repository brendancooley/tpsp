import numpy as np
import imp
import sys

import c_results as results
import c_policies as policies
imp.reload(results)
imp.reload(policies)

location = "local"
size = "mid/"

run_cfact1 = False
run_cfact2 = False
run_cfact3 = False
run_cfact4 = False

results_0 = results.results(location, size)
pecmy_0 = policies.policies(results_0.data, results_0.params, results_0.ROWname, 0)

x_base = np.genfromtxt(results_0.setup.xlhvt_star_path)

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

results_1 = results.results(location, size)
pecmy_1 = policies.policies(results_1.data, results_1.params, results_1.ROWname, 0)

if run_cfact1 == True:
    print("beginning counterfactual 1...")
    xlhvt_prime_1 = results_1.compute_counterfactual(v_500, theta_x, pecmy_1.mzeros, start_with_resto=False, tau_bounds=False, tau_buffer_lower=2.25, tau_buffer_upper=2.25)
    # xlhvt_prime_1 = results_1.compute_counterfactual(v_500, theta_x, pecmy_1.mzeros, start_with_resto=False, tau_bounds=False, tau_buffer_lower=.25, tau_buffer_upper=2.25)
    np.savetxt(results_1.setup.cfct_demilitarization_path + "x.csv", xlhvt_prime_1, delimiter=",")

xlhvt_prime_1 = np.genfromtxt(results_1.setup.cfct_demilitarization_path + "x.csv", delimiter=",")
ge_x_star_1 = pecmy_1.rewrap_xlhvt(xlhvt_prime_1)["ge_x"]
h_1 = pecmy_1.rewrap_xlhvt(xlhvt_prime_1)["h"]

tau_1 = pecmy_1.ecmy.rewrap_ge_dict(ge_x_star_1)["tau_hat"] * pecmy_1.ecmy.tau
X_star_1 = pecmy_1.ecmy.rewrap_ge_dict(ge_x_star_1)["X_hat"] * pecmy_1.ecmy.Xcif  # counterfactual trade flows
Ghat_1 = pecmy_1.G_hat(ge_x_star_1, v_500, 0, all=True)
Uhat_1 = pecmy_1.ecmy.U_hat(pecmy_1.ecmy.rewrap_ge_dict(ge_x_star_1), np.ones(pecmy_1.N))  # consumer welfare under v=1 for all i

np.savetxt(results_1.setup.cfct_demilitarization_path + "tau.csv", tau_1, delimiter=",")
np.savetxt(results_1.setup.cfct_demilitarization_path + "X_prime.csv", X_star_1, delimiter=",")
np.savetxt(results_1.setup.cfct_demilitarization_path + "G_hat.csv", Ghat_1, delimiter=",")
np.savetxt(results_1.setup.cfct_demilitarization_path + "U_hat.csv", Uhat_1, delimiter=",")

### COUNTERFACTUAL 2: MILEX 2030 ###

M2030 = np.genfromtxt(results_0.setup.M2030_path, delimiter=",")

results_2 = results.results(location, size)

CHN_id = np.where(results_2.data["ccodes"]=="CHN")
# USA_id = np.where(results_2.data["ccodes"]=="USA")
# results_2.data["M"][CHN_id] = results_2.data["M"][USA_id]
results_2.data["M"] = M2030

pecmy_2 = policies.policies(results_2.data, results_2.params, results_2.ROWname, 0)

# sv = pecmy_2.estimator_sv(pecmy_2.m, np.mean(pecmy_2.ecmy.tau, axis=1), theta_x)
# sv = pecmy_2.estimator_sv(pecmy_2.m, v_500, theta_x)
sv = x_base

# pecmy_2 = policies.policies(results_2.data, results_2.params, results_2.ROWname, 0, tau_buffer_lower=.75, tau_bounds=True)
# np.reshape(pecmy_2.estimator_bounds(theta_x, v_500, sv)[0:pecmy_2.N**2], (pecmy_2.N, pecmy_2.N)) * pecmy_2.ecmy.tau

if run_cfact2 == True:
    print("beginning counterfactual 2...")
    xlhvt_prime_2 = results_2.compute_counterfactual(v_500, theta_x, pecmy_2.m, sv=sv, tau_bounds=True, ge_ones=False, tau_buffer_lower=1.25, tau_buffer_upper=1.25, start_with_resto=True, proximity_weight_off=True)
    # xlhvt_prime_2 = results_2.compute_counterfactual(v_500, theta_x, pecmy_2.m, sv=sv, tau_bounds=True, ge_ones=False, tau_buffer_lower=.75, tau_buffer_upper=1.25, start_with_resto=True, proximity_weight_off=True)
    np.savetxt(results_2.setup.cfct_china_path + "x.csv", xlhvt_prime_2, delimiter=",")

xlhvt_prime_2 = np.genfromtxt(results_2.setup.cfct_china_path + "x.csv", delimiter=",")
ge_x_star_2 = pecmy_2.rewrap_xlhvt(xlhvt_prime_2)["ge_x"]
h_2 = pecmy_1.rewrap_xlhvt(xlhvt_prime_2)["h"]

tau_2 = pecmy_2.ecmy.rewrap_ge_dict(ge_x_star_2)["tau_hat"] * pecmy_2.ecmy.tau
X_star_2 = pecmy_2.ecmy.rewrap_ge_dict(ge_x_star_2)["X_hat"] * pecmy_2.ecmy.Xcif  # counterfactual trade flows
Ghat_2 = pecmy_2.G_hat(ge_x_star_2, v_500, 0, all=True)
Uhat_2 = pecmy_2.ecmy.U_hat(pecmy_2.ecmy.rewrap_ge_dict(ge_x_star_2), np.ones(pecmy_2.N))  # consumer welfare under v=1 for all i

pp_2 = np.ones((pecmy_2.N, pecmy_2.N))
for i in range(pecmy_2.N):
    if i != pecmy_2.ROW_id:
        pp_2_i = pecmy_2.peace_probs(ge_x_star_2, h_2, i, pecmy_2.m, v_500, theta_dict)[1]
    else:
        pp_2_i = np.ones(pecmy_2.N)
    tick = 0
    for j in range(pecmy_2.N):
        if i != j and j != pecmy_2.ROW_id:
            pp_2[i, j] = pp_2_i[tick]
            tick += 1

np.savetxt(results_2.setup.cfct_china_path + "tau.csv", tau_2, delimiter=",")
np.savetxt(results_2.setup.cfct_china_path + "X_prime.csv", X_star_2, delimiter=",")
np.savetxt(results_2.setup.cfct_china_path + "G_hat.csv", Ghat_2, delimiter=",")
np.savetxt(results_2.setup.cfct_china_path + "U_hat.csv", Uhat_2, delimiter=",")
np.savetxt(results_2.setup.cfct_china_path + "pp.csv", pp_2, delimiter=",")

### COUNTERFACTUAL 3: VALUE OF U.S. MILITARY ###

results_3 = results.results(location, size)

USA_id = np.where(results_3.data["ccodes"]=="USA")
results_3.data["M"][USA_id] = results_3.data["M"][USA_id] / 2

pecmy_3 = policies.policies(results_3.data, results_3.params, results_3.ROWname, 0)

sv = x_base
# sv = pecmy_3.estimator_sv(pecmy_3.m, np.mean(pecmy_3.ecmy.tau, axis=1), theta_x)
# sv = pecmy_3.estimator_sv(pecmy_3.m, v_500, theta_x)

x_catch_path = results_3.setup.cfct_us_path + "x_catch.csv"

if run_cfact3 == True:
    print("beginning counterfactual 3...")
    xlhvt_prime_3 = results_3.compute_counterfactual(v_500, theta_x, pecmy_3.m, sv=sv, tau_bounds=True, ge_ones=False, tau_buffer_lower=1.5, tau_buffer_upper=1.5, start_with_resto=True, proximity_weight_off=True, catch=False, catch_path=x_catch_path)
    np.savetxt(results_3.setup.cfct_us_path + "x.csv", xlhvt_prime_3, delimiter=",")
    # x_catch = np.genfromtxt(x_catch_path, delimiter=",")
    # print(x_catch)
    # pecmy_3.estimator_cons_grad(x_catch, pecmy_3.m)
    # sys.stdout.flush()
    # xlhvt_prime_3 = results_3.compute_counterfactual(v_500, theta_x, pecmy_3.m, sv=x_catch, tau_bounds=False, ge_ones=False, tau_buffer_lower=1.5, tau_buffer_upper=2., start_with_resto=False, proximity_weight_off=False, catch=False)

xlhvt_prime_3 = np.genfromtxt(results_3.setup.cfct_us_path + "x.csv", delimiter=",")
ge_x_star_3 = pecmy_3.rewrap_xlhvt(xlhvt_prime_3)["ge_x"]
h_3 = pecmy_3.rewrap_xlhvt(xlhvt_prime_3)["h"]

tau_3 = pecmy_3.ecmy.rewrap_ge_dict(ge_x_star_3)["tau_hat"] * pecmy_3.ecmy.tau
X_star_3 = pecmy_3.ecmy.rewrap_ge_dict(ge_x_star_3)["X_hat"] * pecmy_3.ecmy.Xcif  # counterfactual trade flows
Ghat_3 = pecmy_3.G_hat(ge_x_star_3, v_500, 0, all=True)
Uhat_3 = pecmy_3.ecmy.U_hat(pecmy_3.ecmy.rewrap_ge_dict(ge_x_star_3), np.ones(pecmy_3.N))  # consumer welfare under v=1 for all i

pp_3 = np.ones((pecmy_3.N, pecmy_3.N))
for i in range(pecmy_3.N):
    if i != pecmy_3.ROW_id:
        pp_3_i = pecmy_3.peace_probs(ge_x_star_3, h_3, i, pecmy_3.m, v_500, theta_dict)[1]
    else:
        pp_3_i = np.ones(pecmy_3.N)
    tick = 0
    for j in range(pecmy_3.N):
        if i != j and j != pecmy_3.ROW_id:
            pp_3[i, j] = pp_3_i[tick]
            tick += 1

np.savetxt(results_3.setup.cfct_us_path + "tau.csv", tau_3, delimiter=",")
np.savetxt(results_3.setup.cfct_us_path + "X_prime.csv", X_star_3, delimiter=",")
np.savetxt(results_3.setup.cfct_us_path + "G_hat.csv", Ghat_3, delimiter=",")
np.savetxt(results_3.setup.cfct_us_path + "U_hat.csv", Uhat_3, delimiter=",")
np.savetxt(results_3.setup.cfct_us_path + "pp.csv", pp_3, delimiter=",")

### COUNTERFACTUAL 4: CHINESE PREFERENCE LIBERALIZATION ###

results_4 = results.results(location, size)

USA_id = np.where(results_4.data["ccodes"]=="USA")
CHN_id = np.where(results_4.data["ccodes"]=="CHN")
v_4 = np.copy(v_500)
v_4[CHN_id] = v_4[USA_id]

pecmy_4 = policies.policies(results_4.data, results_4.params, results_4.ROWname, 0)

sv = x_base

if run_cfact4 == True:
    print("beginning counterfactual 4...")
    xlhvt_prime_4 = results_4.compute_counterfactual(v_4, theta_x, pecmy_4.m, sv=sv, tau_bounds=False, ge_ones=False, tau_buffer_lower=1.75, tau_buffer_upper=1.75, start_with_resto=False, proximity_weight_off=True)
    np.savetxt(results_4.setup.cfct_china_v_path + "x.csv", xlhvt_prime_4, delimiter=",")

xlhvt_prime_4 = np.genfromtxt(results_4.setup.cfct_china_v_path + "x.csv", delimiter=",")
ge_x_star_4 = pecmy_4.rewrap_xlhvt(xlhvt_prime_4)["ge_x"]
h_4 = pecmy_4.rewrap_xlhvt(xlhvt_prime_4)["h"]

tau_4 = pecmy_4.ecmy.rewrap_ge_dict(ge_x_star_4)["tau_hat"] * pecmy_4.ecmy.tau
X_star_4 = pecmy_4.ecmy.rewrap_ge_dict(ge_x_star_4)["X_hat"] * pecmy_4.ecmy.Xcif  # counterfactual trade flows
Ghat_4 = pecmy_4.G_hat(ge_x_star_4, v_4, 0, all=True)
Uhat_4 = pecmy_4.ecmy.U_hat(pecmy_4.ecmy.rewrap_ge_dict(ge_x_star_4), np.ones(pecmy_4.N))  # consumer welfare under v=1 for all i

pp_4 = np.ones((pecmy_4.N, pecmy_4.N))
for i in range(pecmy_4.N):
    if i != pecmy_4.ROW_id:
        pp_4_i = pecmy_4.peace_probs(ge_x_star_4, h_4, i, pecmy_4.m, v_4, theta_dict)[1]
    else:
        pp_4_i = np.ones(pecmy_4.N)
    tick = 0
    for j in range(pecmy_4.N):
        if i != j and j != pecmy_4.ROW_id:
            pp_4[i, j] = pp_4_i[tick]
            tick += 1

np.savetxt(results_4.setup.cfct_china_v_path + "tau.csv", tau_4, delimiter=",")
np.savetxt(results_4.setup.cfct_china_v_path + "X_prime.csv", X_star_4, delimiter=",")
np.savetxt(results_4.setup.cfct_china_v_path + "G_hat.csv", Ghat_4, delimiter=",")
np.savetxt(results_4.setup.cfct_china_v_path + "U_hat.csv", Uhat_4, delimiter=",")
np.savetxt(results_4.setup.cfct_china_v_path + "pp.csv", pp_4, delimiter=",")
