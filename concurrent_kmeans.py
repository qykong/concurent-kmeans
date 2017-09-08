from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from numpy.core.umath_tests import inner1d

MAX_RECORDS_PER_WORKER = 30

################### simulating data related operations ###########################
class database():
    def __init__(self, data):
        self.data = data

    def shape(self):
        return self.data.shape

    def get_records(self, indices):
        return self.data[indices]

# haven't got time to test yet....
DATA = np.genfromtxt('sample.csv', delimiter = ',')
DB = database(DATA)

##################### begin of worker machine part ###################################
# worker function
def worker(fun, indices, centroids = None, K = None, assign = None):
    data = DB.get_records(indices)
    return fun(data, indices, centroids, K, assign)

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
##################### end of worker machine part ###################################

##################### begin of slaver machine part ###################################
def reassign(no_records, centroids, K, workers):
    new_assign = np.zeros(no_records)

    # assign to worker based on capacity
    no_pieces = no_records // MAX_RECORDS_PER_WORKER + 1
    reses = [workers.submit(worker, worker_reassign, \
                       idxs, \
                       centroids, K) for idxs in np.array_split(np.array(range(no_records)), no_pieces)]
    for res in as_completed(reses):
        try:
            tmp_assign, tmp_ids = res.result()
            new_assign[tmp_ids] = tmp_assign
        except Exception as e:
            print(e)

    return new_assign

def recompute(no_records, dims, K, assign, workers):
    new_centroids = np.zeros((K, dims))
    whole_idxs = np.array(range(no_records))

    for k in range(K):
        idxs = np.where(assign == k)
        k_idxs = whole_idxs[idxs]
        # assign to worker based on capacity
        no_pieces = len(k_idxs) // MAX_RECORDS_PER_WORKER + 1
        splitted_idxs = np.array_split(k_idxs, no_pieces)

        reses = [workers.submit(worker, compute_mean, i) for i in splitted_idxs]
        for res in as_completed(reses):
            computed_mean = np.zeros(dims)
            total_no = len(k_idxs)
            try:
                tmp_computed_mean, idxs = res.result()
                computed_mean += (len(idxs) / total_no) * tmp_computed_mean
            except Exception as e:
                print(e)
        new_centroids[k, ] = computed_mean

    return new_centroids

def init_centroids(no_records, K):
    # random select centroids
    indices = np.random.choice(no_records, K, replace = False)
    centroids = DB.get_records(indices)

    return centroids

def kmeans(no_records, dims, K, workers, max_iters = 100, tol = 1e-3):
    n_iter = 1
    converged = False

    centroids = init_centroids(no_records, K)
    assign = np.zeros(no_records)

    while n_iter < max_iters and not converged:
        assign = reassign(no_records, centroids, K, workers)
        new_centroids = recompute(no_records, dims, K, assign, workers)

        diff = dist(centroids, new_centroids).sum()
        converged =  diff <= tol
        centroids = new_centroids

        print('iter %s, diff %s'.format(str(n_iter), str(diff)))
        n_iter += 1

    return assign
##################### end of slaver machine part ###################################

if __name__ == '__main__':
    no_records, dims = DATA.shape()
    K = 10
    no_workers = 5
    # use processes to simulate multipule machines
    with ProcessPoolExecutor(no_workers) as workers:
        print('--------- result -----------')
        print(kmeans(no_records, dims, K, workers))
