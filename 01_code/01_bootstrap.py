import sys
# import threading
import multiprocessing as mp
import logging
import imp
import os
import numpy as np
from scipy import optimize as opt
import copy

import c_results as results
import c_policies as policies

location = sys.argv[1]  # local, hpc
size = sys.argv[2] # mini/, mid/, large/
Mstart = int(sys.argv[3])
Mend = int(sys.argv[4])
# location = "local"
# size = "mid/"

M = 100 # number of bootstrap iterations

results_base = False
results_bootstrap = True
mil_off = True

r_base = results.results(location, size, bootstrap_id=0)

if results_base == True:
    print("results base...")
    r_base.compute_estimates()
    r_base.unravel_estimates()

x_base = np.genfromtxt(r_base.xlhvt_star_path)

def bootstrap_i(id, mil_off=False):
    # r_id = results.results(location, size, sv=x_base, bootstrap=True, bootstrap_id=id)
    r_id = results.results(location, size, bootstrap=True, bootstrap_id=id, mil_off=mil_off)
    if os.path.exists(r_id.xlhvt_star_path):
        print("bootstrap id " + str(id) + " completed, proceeding...")
        sys.stdout.flush()
    else:
        print("beginning bootstrap id " + str(id) + "...")
        sys.stdout.flush()
        r_id.compute_estimates()
    # print("beginning bootstrap id " + str(id) + "...")
    # sys.stdout.flush()
    # r_id.compute_estimates()

if __name__ == '__main__':

    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    if results_bootstrap == True:
        if location == "hpc":
            num_cores = int(os.getenv('SLURM_CPUS_PER_TASK'))
            pool = mp.Pool(num_cores)
        else:
            pool = mp.Pool()
        # for i in range(2, 3):
        for i in range(Mstart, Mend+1):
            pool.apply_async(bootstrap_i, args=(i, mil_off, ))
        pool.close()
        pool.join()

    print("done.")
