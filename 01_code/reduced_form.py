import numpy as np
import imp

import results
import policies
imp.reload(policies)
imp.reload(results)

location = "local"
size = "mid/"

r_base = results.results(location, size, bootstrap_id=0)
x_base = np.genfromtxt(r_base.xlhvt_star_path)

pecmy = policies.policies(r_base.data, r_base.params, r_base.ROWname, 0)

rcv_ft = pecmy.rcv_ft(np.ones(pecmy.x_len), np.ones(pecmy.N))
np.savetxt(r_base.resultsPath + "rcv_ft.csv", rcv_ft, delimiter=",")
