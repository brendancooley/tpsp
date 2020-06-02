import sys
# import threading
import multiprocessing as mp
import logging
import imp
import os
import numpy as np
from scipy import optimize as opt
import copy

import results
import policies
imp.reload(policies)
imp.reload(results)

location = sys.argv[1]  # local, hpc
size = sys.argv[2] # mini/, mid/, large/
Mstart = int(sys.argv[3])
Mend = int(sys.argv[4])
# location = "local"
# size = "mid/"

# mp.cpu_count()

results_base = False
results_bootstrap = True
M = 100 # number of bootstrap iterations

r_base = results.results(location, size, bootstrap_id=0)
# pecmy_base = policies.policies(r_base.data, r_base.params, r_base.ROWname, 0)
# pecmy_base.x_len + pecmy_base.lambda_i_len*pecmy_base.N + pecmy_base.hhat_len*pecmy_base.N
# pecmy_base.ecmy.tau

if results_base == True:
    print("results base...")
    r_base.compute_estimates()
    r_base.unravel_estimates()

x_base = np.genfromtxt(r_base.xlhvt_star_path)
# pecmy = policies.policies(r_base.data, r_base.params, r_base.ROWname)
# x_base_ge_x = pecmy.rewrap_xlhvt(x_base)["ge_x"]
# pecmy.ecmy.tau
# pecmy.ecmy.rewrap_ge_dict(x_base_ge_x)["tau_hat"] * pecmy.ecmy.tau
# # v_base = pecmy.rewrap_xlhvt(x_base)["v"]
# # pecmy.ecmy.rewrap_ge_dict(pecmy.geq_lb(x_base))["tau_hat"] * pecmy.ecmy.tau
# pecmy.ecmy.rewrap_ge_dict(pecmy.rewrap_xlhvt(pecmy.estimator_bounds(pecmy.rewrap_xlhvt(x_base)["theta"], pecmy.rewrap_xlhvt(x_base)["v"], x_base, "lower"))["ge_x"])["tau_hat"] * pecmy.ecmy.tau

def bootstrap_i(id):
    # r_id = results.results(location, size, sv=x_base, bootstrap=True, bootstrap_id=id)
    r_id = results.results(location, size, bootstrap=True, bootstrap_id=id)
    r_id.compute_estimates()

# bootstrap_i(1)
# r_1 = results.results(location, size, sv=x_base, bootstrap=True, bootstrap_id=2)
# pecmy_1 = policies.policies(r_1.data, r_1.params, r_1.ROWname)
# opt.root(pecmy_1.v_upper, x0=np.ones(pecmy_1.N))['x']
# pecmy_1.ecmy.tau
# pecmy_1.ecmy.Y + pecmy_1.ecmy.r_v(pecmy_1.rewrap_xlhvt(x_base)["v"])
# np.maximum([1, -1, 2], 0)
# pecmy_1.rewrap_xlhvt(pecmy_1.update_sv(x_base))["v"]
# pecmy_1.rewrap_xlhvt(x_base)["v"]

# tau_min_mat = copy.deepcopy(pecmy.ecmy.tau)
# np.fill_diagonal(tau_min_mat, 5)
# np.min(tau_min_mat, axis=1) - .25
# np.reshape(np.repeat(np.min(tau_min_mat - pecmy_1.tau_buffer_lower, axis=1), pecmy_1.N), (pecmy_1.N, pecmy_1.N))
# pecmy_1.x_len + pecmy_1.lambda_i_len * 2
# # for i in range(2, 3):
# #     print(i)
# pecmy_1.ecmy.rewrap_ge_dict(pecmy.geq_lb(x_base))["tau_hat"] * pecmy_1.ecmy.tau
# pecmy_1.ecmy.tau
# pecmy_1.ecmy.rewrap_ge_dict(pecmy_1.rewrap_xlhvt(pecmy_1.estimator_bounds(pecmy_1.rewrap_xlhvt(x_base)["theta"], pecmy_1.rewrap_xlhvt(x_base)["v"], x_base, bound= "lower"))["ge_x"])["tau_hat"] * pecmy_1.ecmy.tau


if __name__ == '__main__':

    # processes = list()

    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)
    # for id in range(1, M+1):
    #     t = mp.Process(target=bootstrap_i, args=(id,))
    #     processes.append(t)
    #     t.start()
    #
    # for p in processes:
    #     p.join()

    if results_bootstrap == True:
        if location == "hpc":
            num_cores = int(os.getenv('SLURM_CPUS_PER_TASK'))
            pool = mp.Pool(num_cores)
        else:
            pool = mp.Pool()
        # for i in range(2, 3):
        for i in range(Mstart, Mend):
            pool.apply_async(bootstrap_i, args=(i,))
        pool.close()
        pool.join()

    print("done.")
