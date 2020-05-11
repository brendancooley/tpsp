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
# location = "local"
# size = "mid_RUS/"

# mp.cpu_count()

results_base = False
results_bootstrap = True
M = 100 # number of bootstrap iterations

r_base = results.results(location, size)

if results_base == True:
    print("results base...")
    r_base.compute_estimates()
    r_base.unravel_estimates()

x_base = np.genfromtxt(r_base.xlhvt_star_path)
# pecmy = policies.policies(r_base.data, r_base.params, r_base.ROWname)
# x_base_ge_x = pecmy.rewrap_xlhvt(x_base)["ge_x"]
# pecmy.ecmy.rewrap_ge_dict(x_base_ge_x)["tau_hat"] * pecmy.ecmy.tau

def bootstrap_i(id):
    r_id = results.results(location, size, sv=x_base, bootstrap=True, bootstrap_id=id)
    r_id.compute_estimates()
# bootstrap_i(1)

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
        for i in range(1, M+1):
            pool.apply_async(bootstrap_i, args=(i,))
        pool.close()
        pool.join()

    print("done.")
