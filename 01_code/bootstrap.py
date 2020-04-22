import sys
import threading
import logging
import imp

import results
imp.reload(results)

location = sys.argv[1]  # local, hpc
size = sys.argv[2] # mini/, mid/, large/
# location = "local"
# size = "mid/"

results_base = True
M = 100 # number of bootstrap iterations

if results_base == True:
    print("results base...")
    r_base = results.results(location, size)
    r_base.compute_estimates()
    r_base.unravel_estimates()

def bootstrap_i(id):
    logging.info("Thread starting " + str(id))
    r_id = results.results(location, size, True, id)
    r_id.compute_estimates()
    logging.info("Thread finishing " + str(id))

if __name__ == '__main__':

    threads = list()
