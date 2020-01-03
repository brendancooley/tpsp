import numpy as np
import multiprocessing as mp
import scipy.stats as stats

print("Number of processors: ", mp.cpu_count())

b = []
b.append([1, 1, 1])
b.append([2, 2, 2])

np.mean(b, axis=0)

X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

X * np.array([1, .5, .33])
np.sum(X, axis=1)

sigma_epsilon = 10
stats.norm.pdf(1, loc=0, scale=sigma_epsilon)
stats.norm.pdf(1/sigma_epsilon, loc=0, scale=1)
