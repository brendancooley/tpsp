import sys
# import threading
import multiprocessing as mp
import logging
import imp

import results
import policies
imp.reload(results)

location = sys.argv[1]  # local, hpc
size = sys.argv[2] # mini/, mid/, large/
# location = "local"
# size = "mid_RUS/"

# mp.cpu_count()

results_base = False
M = 100 # number of bootstrap iterations

if results_base == True:
    print("results base...")
    r_base = results.results(location, size)
    r_base.compute_estimates()
    r_base.unravel_estimates()

def bootstrap_i(id):
    r_id = results.results(location, size, True, id)
    r_id.compute_estimates()

# testing
# id = 1
# r_test = results.results(location, size, True, id)
# p_test = policies.policies(r_test.data, r_test.params, r_test.ROWname)
# bootstrap_i(id)

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

    pool = mp.Pool() #use all available cores, otherwise specify the number you want as an argument
    for i in range(1, M+1):
        pool.apply_async(bootstrap_i, args=(i,))
    pool.close()
    pool.join()

    print("done.")
