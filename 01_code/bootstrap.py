import sys
import threading
import logging
import imp

import results
import policies
imp.reload(results)

location = sys.argv[1]  # local, hpc
size = sys.argv[2] # mini/, mid/, large/
location = "local"
size = "mid_RUS/"

results_base = False
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

# testing
# id = 1
# r_test = results.results(location, size, True, id)
# p_test = policies.policies(r_test.data, r_test.params, r_test.ROWname)
# bootstrap_i(id)

if __name__ == '__main__':

    threads = list()

    for id in range(M):
        f = threading.Thread(target=bootstrap_i, args=(id, ))
        threads.append(f)
        f.start()

    for index, thread in enumerate(threads):
        logging.info("Main    : before joining thread %d.", index)
        thread.join()
        logging.info("Main    : thread %d done", index)

    print("done.")
