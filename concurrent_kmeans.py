from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numpy.core.umath_tests import inner1d

def worker(fun, labels, *kargs):
    # TODO: getting data
    return fun(kargs)

# simulating data related operations
class database():
    def __init__(self, data):
        self.data = data

    def shape():
        return self.data.shape

    def get_records(indices):
        pass

# calculation utility functions
def norm(A):
    return np.linalg.norm(A, ord = 2, axis = -1)

def dist(A, b):
    return 1. - (1.0 * inner1d(A, b)) / (norm(A) * norm(b))

# kmeans algo part
def compute_mean(data, indices):
	return (np.mean(data, axis = 0), indices)

def reassign(data, indices, centroids, K):
    n, d = data.shape
    new_assign = np.zeros(n)
    dists = np.zeros((n, K))

    for k in range(K):
        dists[:, k] = dist(data, centroids[k])

    new_assign = np.argmin(dists, axis = 1)
    return (new_assign, indices)

def cost(data, indices, centroids):

