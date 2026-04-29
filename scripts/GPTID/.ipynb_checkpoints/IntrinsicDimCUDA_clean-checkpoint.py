import numpy as np
from threading import Thread
from scipy.spatial.distance import cdist
import torch

import math

MINIMAL_CLOUD = 80
USE_64 = True
if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf) ** 0.5

    
def process_string(sss):
    return sss.replace('\n', ' ').replace('  ', ' ')


def slope_estimation(x, y):
    N = len(x)
    slope = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    return slope


def estimation_bias(x, y):
    slope = slope_estimation(x, y)
    
    return np.mean(y) - np.mean(x) * slope


def phd_estimation(x, y):
    bias = estimation_bias(x, y)
    estimators = []
    for i in range(len(x)):
        estimators.append(x[i] / (bias - y[i] + x[i]))

    return np.median(estimators)


class PH():
    def __init__(self, use_cuda=False, use_my_phd_estimation=True, distance_matrix=False):
        self.use_cuda = use_cuda
        self.use_my_phd_estimation = use_my_phd_estimation
        self.distance_matrix = distance_matrix

    def fit_transform(self, X, dist=False):
        mx_points = X.shape[0]
        mn_points = 10
        step = max(1, ( mx_points - mn_points ) // 10)
        self.distance_matrix = dist

        return self.calculate_ph_dim(X, min_points=mn_points, max_points=mx_points, point_jump=step)

    def sample_W(self, W, nSamples):
        '''
        Sample <<nSamples>> points from the cloud <<W>>
        '''
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        if not self.distance_matrix:
            return W[random_indices]
        return W[random_indices][:, random_indices]

    def prim_tree(self, adj_matrix, power=1.0):
        '''
        Computation of H0S for a point cloud with distance matrix <<adj_matrix>> by using Prim's algorithm 
        for minimal spanning tree
        '''
        infty = np.max(adj_matrix) + 1.0
    
        dst = np.ones(adj_matrix.shape[0]) * infty
        visited = np.zeros(adj_matrix.shape[0], dtype=bool)
        ancestor = -np.ones(adj_matrix.shape[0], dtype=int)
#         print("ancestor.shape:", ancestor.shape)
#         print("dst.shape:", dst.shape)
#         print("adj_matrix.shape:", adj_matrix.shape)

        v, s = 0, 0.0
        for i in range(adj_matrix.shape[0] - 1):
            visited[v] = 1
            ancestor[dst > adj_matrix[v]] = v
            dst = np.minimum(dst, adj_matrix[v])
            dst[visited] = infty
            
            v = np.argmin(dst)
            
            s += adj_matrix[v][ancestor[v]] ** power
        return s.item()

    def calculate_ph_dim(self, W, min_points, max_points, point_jump, alpha=1.0, restarts=7, resamples=7):
        '''
        Estimation of the intrinsic (upper-box) dimension of the given point cloud W.
        Parameters:
        
        min_points --- size of minimal subsample to draw
        max_points --- size of maximal subsample to draw
        point_jump --- size of step between subsamples
        restarts --- number of iterations at each sampling size
        print_error -- to print or not computational error
        '''
        max_points = W.shape[0]
#        print("W.shape:", W.shape)

        m_candidates = []
        for i in range(restarts): 
            test_n = range(min_points, max_points, point_jump)
            lengths = []

            for n in test_n:
                reruns = np.ones(resamples)
                for i in range(resamples):
                    tmp = self.sample_W(W, n)
                    if not self.distance_matrix:
                        reruns[i] = self.prim_tree(cdist(tmp, tmp), power=alpha)
                        if self.use_cuda:
                            reruns[i] = self.prim_tree(pairwise_distances(tmp), power=alpha)
                    else:
                        reruns[i] = self.prim_tree(tmp, power=alpha)


                lengths.append(np.median(reruns))

            lengths = np.array(lengths)
            x = np.log(np.array(list(test_n)))
            y = np.log(lengths)

#            print("x = ", x, "y = ", y)

            N = len(x)

            result = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
            if not np.isnan(result):
                m_candidates.append(result)
                
        if len(m_candidates) > 0:
            m = np.median(m_candidates)
        else:
            m = 0
        return alpha / (1 - m)