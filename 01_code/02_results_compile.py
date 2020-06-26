import numpy as np
import imp
import os
import copy
import sys

import c_results as results
import c_policies as policies
imp.reload(results)

location = "local"
size = "mid/"
M = 250

est_dict = dict()
est_dict["alpha1"] = []
est_dict["alpha2"] = []
est_dict["gamma"] = []
est_dict["v"] = []
est_dict["rcv"] = []
est_dict["peace_probs"] = []
est_dict["tau"] = []
est_dict["Ghat"] = []
est_dict["Uhat1"] = []

est_dict_mil_off = copy.deepcopy(est_dict)

for i in range(1, M+1):
    print(i)
    r_i = results.results(location, size, bootstrap=True, bootstrap_id=i)
    if os.path.isfile(r_i.setup.xlhvt_star_path):
        r_i.unravel_estimates(est_dict)
    r_i_mil_off = results.results(location, size, bootstrap=True, bootstrap_id=i, mil_off=True)
    if os.path.isfile(r_i_mil_off.setup.xlhvt_star_path):
        r_i_mil_off.unravel_estimates(est_dict_mil_off)
    sys.stdout.flush()

quantiles = dict()
quantiles_mil_off = dict()
for i in est_dict.keys():
    if i in ["v", "rcv", "peace_probs", "tau", "G_hat", "U_hat1"]:
        quantiles[i] = np.quantile(np.array(est_dict[i]), [.025, .5, .975], axis=0)
        quantiles_mil_off[i] = np.quantile(np.array(est_dict_mil_off[i]), [.025, .5, .975], axis=0)
    else:
        quantiles[i] = np.quantile(est_dict[i], [.025, .5, .975])
        quantiles_mil_off[i] = np.quantile(est_dict_mil_off[i], [.025, .5, .975])

r_base = results.results(location, size)
np.savetxt(r_base.setup.quantiles_v_path, quantiles["v"], delimiter=",")
np.savetxt(r_base.setup.quantiles_gamma_path, quantiles["gamma"], delimiter=",")
np.savetxt(r_base.setup.quantiles_alpha1_path, quantiles["alpha1"], delimiter=",")
np.savetxt(r_base.setup.quantiles_alpha2_path, quantiles["alpha2"], delimiter=",")
np.savetxt(r_base.setup.quantiles_rcv_path, quantiles["rcv"], delimiter=",")
np.savetxt(r_base.setup.quantiles_peace_probs_path, quantiles["peace_probs"], delimiter=",")
np.savetxt(r_base.setup.quantiles_tau_path, quantiles["tau"], delimiter=",")
np.savetxt(r_base.setup.quantiles_Ghat_path, quantiles["Ghat"], delimiter=",")
np.savetxt(r_base.setup.quantiles_Uhat1_path, quantiles["Uhat1"], delimiter=",")

r_base_mil_off = results.results(location, size, mil_off=True)
np.savetxt(r_base_mil_off.setup.quantiles_v_path, quantiles_mil_off["v"], delimiter=",")
