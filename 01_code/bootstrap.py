import sys

import results

# location = sys.argv[1]  # local, hpc
# size = sys.argv[2] # mini/, mid/, large/
location = "local"
size = "mid/"

results_base = True

if results_base == True:
    r_base = results(location, size)
    r_base.compute_estimates()
    r_base.unravel_estimates()
