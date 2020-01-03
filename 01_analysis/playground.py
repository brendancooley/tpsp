import numpy as np
import multiprocessing as mp

print("Number of processors: ", mp.cpu_count())

b = []
b.append([1, 1, 1])
b.append([2, 2, 2])

np.mean(b, axis=0)

X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

X * np.array([1, .5, .33])
np.sum(X, axis=1)
