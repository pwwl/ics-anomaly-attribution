"""
    LEMNA implementation imported from original authors: https://github.com/yuyay/pygflasso
"""

import numpy as np 
from sklearn.base import RegressorMixin, BaseEstimator


class MultiTaskGFLasso(RegressorMixin, BaseEstimator):
    """
    Graph-Guided Fused Lasso
    """
    def __init__(
        self, G, max_iter=50, lamb=1.0, gamma=1.0, eps=1.0, tol=10**-4,
        nobias=False, verbose=False, 
    ):
        self.G = G  # Graph structure of tasks
        self.max_iter = max_iter
        self.lamb = lamb  # strength of l1 regularizer
        self.gamma = gamma  # strength of fusion regularizer
        self.eps = eps  
        self.tol = tol
        self.nobias = nobias
        self.verbose = verbose

    def fit(self, X, y):
        if not self.nobias:
            X = self._add_bias(X)
        n, d = X.shape
        n_tasks = y.shape[1]
        n_edges = np.nonzero(self.G)[0].shape[0] # TODO: sparsify

        # Initialize
        H = np.zeros((n_tasks, n_edges))
        for k in range(n_tasks):
            for e, (m, l) in enumerate(zip(*np.nonzero(self.G))):
                if k == m:
                    H[k, e] = np.abs(self.G[m, l])
                elif k == l:
                    H[k, e] = -np.sign(self.G[m, l]) * np.abs(self.G[m, l])
        I = np.identity(n_tasks)
        C = np.concatenate((self.lamb * I, self.gamma * H), axis=1)
        D = 0.5 * d * (n_tasks + n_edges)
        d_k = np.sum(self.G**2, axis=1)
        mu = self.eps / (2. * D)
        ev, _ = np.linalg.eig(np.dot(X.T, X))
        Lu = np.max(ev) + (self.lamb**2 + 2 * self.gamma**2 * np.max(d_k)) / mu

        # Training
        W_prev = np.zeros((d, n_tasks))
        B = np.ones((d, n_tasks))
        Z_prev = 0.
        coefdiff_history = []
        loss_history = []
        for t in range(self.max_iter):
            # Compute (11)
            A = _shrinkage_operator(np.dot(W_prev, C) / mu)
            fb = np.dot(X.T, (np.dot(X, W_prev) - y)) + np.dot(A, C.T)

            # Gradient descent step
            B = _soft_threshold(W_prev - fb / Lu, self.lamb / Lu)
            
            # Set Z
            Z = Z_prev - 0.5 * (t + 1) * fb / Lu 

            # Set next W_prev
            W_prev = ((t + 1) / (t + 3)) * B + (2 / (t + 3)) * Z 

            # Check termination condition
            loss = self._objective(X, y, B)
            coefdiff = np.mean(np.abs(B - B_prev)) if t > 0 else np.inf
            loss_history.append(loss)
            coefdiff_history.append(coefdiff)
            if self.verbose:
                print("Iter {0:>3}: diff = {1:e}, loss = {2:e}".format(
                    t + 1, coefdiff, loss))
            if coefdiff <= self.tol:
                break
            B_prev = B
            Z_prev = Z
        
        self.coef_ = B.T[:, :-1] if not self.nobias else B.T
        self.intercept_ = B.T[:, -1] if not self.nobias else np.zeros(n_tasks)
        self.coefdiff_history_ = coefdiff_history
        self.loss_history_ = loss_history
        return self

    def predict(self, X):
        y_pred = np.dot(X, self.coef_.T) + self.intercept_
        return y_pred

    def _add_bias(self, X):
        b = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate((X, b), axis=1)

    def _objective(self, X, y, B):
        loss = 0.5 * np.sum((y - np.dot(X, B))**2)
        loss += self.lamb * np.linalg.norm(B, ord=1)
        for e, (m, l) in enumerate(zip(*np.nonzero(self.G))):
            r = self.G[m, l]
            s = np.sign(r)
            loss += self.gamma * np.abs(r) * np.sum(np.abs(B[:, m] - s * B[:, l]))
        return loss


def _shrinkage_operator(array):
    array = np.where(array >= 1., 1., array)  # if x >= 1
    array = np.where(array <= -1., -1., array)  # if x <= -1
    return array


def _soft_threshold(array, lamb):
    array_new = np.zeros_like(array)
    array_new[np.where(array > lamb)] = array[np.where(array > lamb)] - lamb
    array_new[np.where(array < -lamb)] = array[np.where(array < -lamb)] + lamb
    return array_new
