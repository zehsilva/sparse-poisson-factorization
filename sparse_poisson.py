"""

Poisson Matrix Factorization using sparse representation of input matrix by: 2017-11-24 Eliezer de Souza da Silva <eliezer.souza.silva@ntnu.no>
Modification of a code created by: 2014-03-25 02:06:52 by Dawen Liang <dliang@ee.columbia.edu>



"""

import sys
import numpy as np
from scipy import special
import numpy_indexed as npi


from sklearn.base import BaseEstimator, TransformerMixin

class PoissonMF(BaseEstimator, TransformerMixin):
    """ Poisson matrix factorization with batch inference and sparse input matrix and internal representation """
    def __init__(self, n_components=100, max_iter=100, tol=0.0005,
                 smoothness=100, random_state=None, verbose=False,allone=False,
                 **kwargs):
        """ Poisson matrix factorization

        Arguments
        ---------
        n_components : int
            Number of latent components

        max_iter : int
            Maximal number of iterations to perform

        tol : float
            The threshold on the increase of the objective to stop the
            iteration

        smoothness : int
            Smoothness on the initialization variational parameters

        random_state : int or RandomState
            Pseudo random number generator used for sampling

        verbose : bool
            Whether to show progress during model fitting

        **kwargs: dict
            Model hyperparameters: theta_a, theta_b, beta_a, beta_b
        """
        self.allone=allone
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a1 = float(kwargs.get('theta_a', 0.1))
        self.a2 = float(kwargs.get('theta_b', 0.1))
        self.b1 = float(kwargs.get('beta_a', 0.1))
        self.b2 = float(kwargs.get('beta_b', 0.1))

    def _init_components(self, n_rows):
        # variational parameters for beta
        self.gamma_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_rows, self.n_components))
        self.rho_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_rows, self.n_components))
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def set_components(self, shape, rate):
        '''Set the latent components from variational parameters.

        Parameters
        ----------
        shape : numpy-array, shape (n_components, n_feats)
            Shape parameters for the variational distribution

        rate : numpy-array, shape (n_components, n_feats)
            Rate parameters for the variational distribution

        Returns
        -------
        self : object
            Return the instance itself.
        '''

        self.gamma_b, self.rho_b = shape, rate
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)
        return self

    def _init_weights(self, n_cols):
        # variational parameters for theta
        self.gamma_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_cols, self.n_components))
        self.rho_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_cols, self.n_components))
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)
        self.c = 1. / np.mean(self.Et)

    def fit(self, X):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : array-like, shape (n_examples, 3)

            Training data.

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        X_new=np.zeros(shape=X.shape,dtype=int)
        if self.allone:
            X_new[:, -1]=1
        else:
            X_new[:, -1]=X[:,-1] ## copy the last column
        unique_rows= np.unique(X[:,0])
        unique_cols= np.unique(X[:,1])
        d_rows = dict(zip(unique_rows,range(len(unique_rows))))
        d_cols = dict(zip(unique_cols, range(len(unique_cols))))
        X_new[:, 0] = np.array([d_rows[x] for x in X[:, 0]],dtype=np.int64)
        X_new[:, 1] = np.array([d_cols[x] for x in X[:, 1]],dtype=np.int64)
        self.n_rows = np.max(X_new[:,0])+1
        self.n_cols = np.max(X_new[:,1])+1
        if self.verbose:
            print "cols=",self.n_cols
            print "rows=",self.n_rows
        self.row_index = X_new[:,0]
        self.cols_index = X_new[:,1]
        self.vals_vec = X_new[:,2]
        self._init_components(self.n_rows) #beta
        self._init_weights(self.n_cols) #theta
        self._update(X_new)
        return self

    def transform(self, X, attr=None):
        '''Encode the data as a linear combination of the latent components.

        TODO
        '''
        return 1
    def _update_phi(self,X):
        self.phi_var = np.zeros((X.shape[0], self.n_components))
        self.phi_var = np.add(self.phi_var, np.exp(self.Elogb[self.row_index, :]))
        self.phi_var = np.add(self.phi_var, np.exp(self.Elogt[self.cols_index, :]))
        self.phi_var = np.divide(self.phi_var, np.sum(self.phi_var, axis=1)[:, np.newaxis])
        self.phi_var =self.vals_vec[:,np.newaxis]*self.phi_var

    def _update(self, X, update_beta=True):
        # alternating between update latent components and weights
        old_bd = -np.inf

        for i in xrange(self.max_iter):
            self._update_phi(X)

            self._update_theta(X)
            if update_beta:
                self._update_phi(X)
                self._update_beta(X)
            bound = self._bound(X)
            improvement = (bound - old_bd) / abs(old_bd)
            if self.verbose:
                sys.stdout.write('\r\tAfter ITERATION: %d\tObjective: %.2f\t'
                                 'Old objective: %.2f\t'
                                 'Improvement: %.5f' % (i, bound, old_bd,
                                                        improvement))
                sys.stdout.flush()
            if improvement < self.tol:
                break
            old_bd = bound
        if self.verbose:
            sys.stdout.write('\n')
        pass

    def _update_theta(self, X):
        #print "t_gamma 0", self.gamma_t.shape
        #print "t_rho 0", self.rho_t.shape
        self.gamma_t = self.a1 + npi.group_by(self.cols_index).sum(self.phi_var)[1]
        self.rho_t = self.a2  + np.sum(self.Eb, axis=0, keepdims=True)
        #print "t_gamma 1", self.gamma_t.shape
        #print "t_rho 1", self.rho_t.shape
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)
        #self.c = 1. / np.mean(self.Et)

    def _update_beta(self, X):
        #print "b_gamma 0", self.gamma_b.shape
        #print "b_rho 0", self.rho_b.shape
        self.gamma_b = self.b1 + npi.group_by(self.row_index).sum(self.phi_var)[1]
        self.rho_b = self.b2 + np.sum(self.Et, axis=0, keepdims=True)
        #print "b_gamma 1", self.gamma_b.shape
        #print "b_rho 1", self.rho_b.shape
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def _bound(self, X):
        bound = np.sum(self.phi_var*(self.Elogt[self.cols_index, :]+self.Elogb[self.row_index, :]))
        bound -= np.sum(self.phi_var*(np.log(self.phi_var)-np.log(X[:,2]).reshape(X.shape[0],1)))
        bound -= np.sum(np.inner(self.Eb,self.Et))
        bound += _gamma_term(self.a1, self.a2 ,
                             self.gamma_t, self.rho_t,
                             self.Et, self.Elogt)
        bound += _gamma_term(self.b1, self.b2, self.gamma_b, self.rho_b,
                             self.Eb, self.Elogb)
        return bound


def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    #beta=beta.reshape((beta.shape[0], 1))
    return (alpha / beta, special.psi(alpha) - np.log(beta))

def _compute_entropy(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute Entropy[x]
    '''
    #beta=beta.reshape((beta.shape[0], 1))
    return alpha+(1-alpha)*special.psi(alpha) - np.log(beta)+special.gammaln(alpha)

def _gamma_term(a, b, shape, rate, Ex, Elogx):
    return np.sum((a - shape) * Elogx - (b - rate) * Ex +
                  (special.gammaln(shape) - shape * np.log(rate)))

def _sum_product_newaxis1(auxvar, data, axis=1):
    return np.sum(auxvar * data[np.newaxis, :, :], axis=axis)
