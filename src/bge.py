import numpy as np
import math
from scipy.special import gammaln, multigammaln
from numpy.linalg import slogdet

class BGe:
    """Class implementing BGe score computations (for leaf node distributions). See Kuipers et al. (2014)"""
    def __init__(self, d, alpha_u=None, alpha_w=None, T=None, nu=None, prt=False):
        self.d = d
        self.alpha_u = alpha_u
        self.alpha_w = alpha_w
        self.T = T
        self.nu = nu
        self.R = None
        self.N = None
        if alpha_u is None:
            self.alpha_u = 1
        if alpha_w is None:
            self.alpha_w = d + 2
        if T is None:
            self.T = np.eye(d)
        if nu is None:
            self.nu = np.zeros(d)
        self.prt = prt

    def calc_R(self, X):
        """Calculates R matrix given data.

        Args:
            X (np.array): (batch, d) training data

        Returns:
            R (np.array): R (posterior) matrix
        """
        N = X.shape[0]
        X_bar = np.mean(X, axis=0)
        assert ((np.dot((X - X_bar).T, (X - X_bar))).shape == (self.d, self.d))

        R = self.T + np.dot((X - X_bar).T, (X - X_bar)) + \
            (N * self.alpha_u) / (N + self.alpha_u) * np.outer(self.nu - X_bar, self.nu - X_bar)
        self.R = R
        self.N = N
        return R

    def local_score(self, node, parents):
    
        if self.R is None or self.N is None:
            raise ValueError("BGe.local_score: self.R / self.N not set.")

        parents = np.asarray(parents, dtype=int)
        G_i = np.zeros(self.d, dtype=int)
        G_i[parents] = 1

        return float(self._mll_per_variable(node, G_i, self.R, self.N))
        
    def _mll_per_variable(self, i, G_i, R, N):
        """Computes marginal likelihood of data for a given variable and set of parents.

        Args:
            i (int): variable index
            G_i (np.array): set of parents of variable i
            R (np.array): R matrix (computed using calc_R once for the entire dataset)
            N (int): number of observations in training data

        Returns:
            log_diff (float): marginal log likelihood
        """
        parent_mask = G_i.astype(bool)
        parent_child_mask = np.copy(parent_mask)
        parent_child_mask[i] = 1

        sign, logdet = slogdet(self.T[np.ix_(parent_child_mask, parent_child_mask)])
        log_det_T_num = logdet  # * sign
        sign, logdet = slogdet(self.T[np.ix_(parent_mask, parent_mask)])
        log_det_T_den = logdet  # * sign
        sign, logdet = slogdet(R[np.ix_(parent_child_mask, parent_child_mask)])
        log_det_R_num = logdet  # * sign
        sign, logdet = slogdet(R[np.ix_(parent_mask, parent_mask)])
        log_det_R_den = logdet  # * sign

        l = np.sum(parent_mask)

        const_log_prefactor = 0.5 * (math.log(self.alpha_u) - math.log(N + self.alpha_u))
        log_prefactor = (
                const_log_prefactor
                + gammaln(0.5 * (N + self.alpha_w - self.d + l + 1))
                - gammaln(0.5 * (self.alpha_w - self.d + l + 1))
                - (0.5 * N) * math.log(math.pi))

        # this is equal to (better way of computing) log_num - log_den
        log_diff = 0.5 * (self.alpha_w - self.d + (l + 1)) * log_det_T_num + \
                  -0.5 * (N + self.alpha_w - self.d + (l + 1)) * log_det_R_num + \
                  - 0.5 * (self.alpha_w - self.d + l) * log_det_T_den + \
                  + 0.5 * (N + self.alpha_w - self.d + l) * log_det_R_den + \
                  log_prefactor

        return log_diff

    def mll(self, G, X):
        """Computes marginal log-likelihood of data given graph.

        Args:
            G (np.array): (d, d) graph
            X (np.array): (batch, d) training data

        Returns:
            tot (float): marginal log-likelihood
        """
        assert (X.shape[1] == self.d)
        N = X.shape[0]

        R = self.calc_R(X)
        tot = 0
        for i in range(self.d):
            tot += self._mll_per_variable(i, G[:, i], R, N)
        return tot
