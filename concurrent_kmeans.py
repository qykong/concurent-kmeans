from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numpy.core.umath_tests import inner1d

MAX_RECORDS_PER_WORKER = 30

def worker(fun, indices, centroids, K, assign = None):
    # TODO: getting data
    return fun(indices, centroids, K, assign)

# simulating data related operations
class database():
    def __init__(self, data):
        self.data = data

    def shape(self):
        return self.data.shape

    def get_records(self, indices):
        pass

DB = database()

# calculation utility functions
def norm(A):
    return np.linalg.norm(A, ord = 2, axis = -1)

def dist(A, b):
    return 1. - (1.0 * inner1d(A, b)) / (norm(A) * norm(b))

# kmeans algo part for worker
def compute_mean(data, indices, *kargs):
	return (np.mean(data, axis = 0), indices)

def worker_reassign(data, indices, centroids, K, assign):
    n, d = data.shape
    new_assign = np.zeros(n)
    dists = np.zeros((n, K))

    for k in range(K):
        dists[:, k] = dist(data, centroids[k])

    new_assign = np.argmin(dists, axis = 1)
    return (new_assign, indices)

def worker_cost(data, indices, centroids, K, assign):
    costs = np.zeros(K)

    for k in range(K):
        idxs = np.where(assign == k)
        costs[k] = dist(data[idxs], centroids[k]).sum()

    return costs

# slaver part
def reassign(no_records, centroids, K, workers):
    no_pieces = no_records // MAX_RECORDS_PER_WORKER + 1

def init_centroids(no_records, K):
    # random select centroids
    indices = np.random.choice(no_records, K, replace = False)
    centroids = DB.get_records(indices)

    return centroids

def kmeans(no_records, K, max_iters = 100, tol = 1e-3, workers):
    n_iter = 0
    converged = False

    centroids = init_centroids(no_records, K)
    assign = reassign(no_records, centroids, K, workers)
