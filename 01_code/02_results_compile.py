import numpy as np
import imp
import os

import c_results as results
import c_policies as policies

location = "local"
size = "mid/"
M = 100

est_dict = dict()
est_dict["alpha1"] = []
est_dict["alpha2"] = []
est_dict["gamma"] = []
est_dict["v"] = []

for i in range(1, M+1):
    print(i)
    r_i = results.results(location, size, bootstrap=True, bootstrap_id=i)
    if os.path.isfile(r_i.xlhvt_star_path):
        r_i.unravel_estimates(est_dict)

quantiles = dict()
for i in est_dict.keys():
    if i == "v":
        quantiles[i] = np.quantile(np.array(est_dict[i]), [.025, .5, .975], axis=0)
    else:
        quantiles[i] = np.quantile(est_dict[i], [.025, .5, .975])

r_base = results.results(location, size)

for i in quantiles.keys():
    np.savetxt(r_base.estimatesPath + "quantiles_" + i + ".csv", quantiles[i], delimiter=",")
