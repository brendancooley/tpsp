import autograd.numpy as np
import scipy.stats as stats
import math

def find_nearest(array, value):
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def which_nearest(array, value):
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# find_nearest(np.arange(0, 1.1, .1), .54)

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return(w)

def mean_truncnorm(lb, sigma, ub=np.infty, mu=0):
    phi_lb = stats.norm.pdf((lb-mu)/sigma, loc=mu, scale=1)
    phi_ub = stats.norm.pdf((ub-mu)/sigma, loc=mu, scale=1)
    Phi_lb = stats.norm.cdf((lb-mu)/sigma, loc=mu, scale=1)
    Phi_ub = stats.norm.cdf((ub-mu)/sigma, loc=mu, scale=1)
    return(mu + (phi_lb - phi_ub) / (Phi_ub - Phi_lb) * sigma)

# lb = 1
# sigma = 1
# mu = 0
# mean_truncnorm(lb, sigma)
