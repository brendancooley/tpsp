import numpy as np
import imp

import c_results as results
import c_policies as policies

location = "local"
size = "mid/"

r_base = results.results(location, size, bootstrap_id=0)
x_base = np.genfromtxt(r_base.setup.xlhvt_star_path)

pecmy = policies.policies(r_base.data, r_base.params, r_base.ROWname, 0)

rcv_ft = pecmy.rcv_ft(np.ones(pecmy.x_len), np.ones(pecmy.N))
np.savetxt(r_base.setup.rcv_ft_path, rcv_ft, delimiter=",")
