"""

Poisson Tensor Factorization using sparse representation of input tensor by: 2017-11-24 Eliezer de Souza da Silva <eliezer.souza.silva@ntnu.no>
Modification of a code created in: 2014-03-25 02:06:52 by Dawen Liang <dliang@ee.columbia.edu>

"""

import sys
import numpy as np
from scipy import special
import numpy_indexed as npi


from sklearn.base import BaseEstimator, TransformerMixin


class PoissonTF(BaseEstimator, TransformerMixin):
    """ Poisson tensor factorization with batch inference """
    def __init__(self, n_components=100, max_iter=100, tol=0.0005,
                 smoothness=100, random_state=None, verbose=False,
                 **kwargs):
        """ Poisson tensor factorization

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
            Model hyperparameters
        """

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
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))

    def _init_components(self, n_rows):
        # variational parameters modes
        self.mode_sizes = n_rows
        self.gamma_b=[]
        self.rho_b=[]
        self.Eb=[]
        self.Elogb=[]
        self.partial_sums=[]
        self.gamma_lambda=np.random.gamma(self.smoothness, 1. / self.smoothness,
                                  size=(1, self.n_components))
        self.rho_lambda=np.random.gamma(self.smoothness, 1. / self.smoothness,
                                  size=(1, self.n_components))
        self.Elambda,self.Eloglambda=_compute_expectations(self.gamma_lambda, self.rho_lambda)
        for mode_n in n_rows:
            gamma_b=self.smoothness \
                * np.random.gamma(self.smoothness, 1. / self.smoothness,
                                  size=(mode_n, self.n_components))
            rho_b = self.smoothness \
                * np.random.gamma(self.smoothness, 1. / self.smoothness,
                                  size=(mode_n, self.n_components))
            self.gamma_b.append(gamma_b)
            self.rho_b.append(rho_b)
            tempE, tempElog = _compute_expectations(gamma_b, rho_b)
            self.partial_sums.append(np.sum(tempE,axis=0,keepdims=True)) ### shape=(1,self.n_components)
            self.Eb.append(tempE)
            self.Elogb.append(tempElog)

    def set_components(self, shape, rate):
        '''Set the latent components from variational parameters.

        Parameters
        ----------
        shape : list of numpy-array, shape (n_items_mode, n_components)
            Shape parameters for the variational distribution

        rate : list of numpy-array, shape (n_items_mode, n_components)
            Rate parameters for the variational distribution

        Returns
        -------
        self : object
            Return the instance itself.
        '''

        self.gamma_b, self.rho_b = shape, rate
        self.Eb, self.Elogb = shape,rate
        for s,r,i in zip(shape,rate,range(len(shape))):
            self.Eb[i], self.Elogb[i] = _compute_expectations(s, r)
        return self

    def fit(self, X):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : sparse tensor array-like, shape (n_examples, n_modes+1)

            Training data.
            [[X_mode_1,...,X_mode_n,V],
            ....]
            
            V[X_mode_1,...,X_mode]] is the value indexed by [X_mode_1,...,X_mode_n] in a dense tensor array

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        self.n_ids_mode = []
        self.unique_id_mode=[]
        X_new=np.zeros(shape=X.shape,dtype=int)
        X_new[:, -1]=X[:,-1] ## copy the last column
        for mode in range(X.shape[1]-1): ## the last column is the value, is not a mode
            unique_ids= np.unique(X[:,mode])
            self.unique_id_mode.append(unique_ids)
            d_ids = dict(zip(unique_ids,range(len(unique_ids))))
            X_new[:, mode] = np.array([d_ids[x] for x in X[:, mode]])
            self.n_ids_mode.append( np.max(X_new[:,mode])+1 )
            if self.verbose:
                print("n_ids in mode "+str(mode)+"= "+str(self.n_ids_mode[mode]))
        self._init_components(self.n_ids_mode)
        self._update(X_new)
        return self

    def transform(self, X, attr=None):
        '''Encode the data as a linear combination of the latent components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)

        attr: string
            The name of attribute, default 'Eb'. Can be changed to Elogb to
            obtain E_q[log beta] as transformed data.

        Returns
        -------
        X_new : array-like, shape(n_samples, n_filters)
            Transformed data, as specified by attr.
        '''
        '''

        if not hasattr(self, 'Eb'):
            raise ValueError('There are no pre-trained components.')
        n_samples, n_feats = X.shape
        if n_feats != self.Eb.shape[1]:
            raise ValueError('The dimension of the transformed data '
                             'does not match with the existing components.')
        if attr is None:
            attr = 'Et'
        self._init_weights(n_samples)
        self._update(X, update_beta=False)
        return getattr(self, attr)
        '''
    def _update(self, X, update_beta=True):
        # alternating between update latent components and weights
        old_bd = -np.inf
        self._update_phi(X)

        for i in xrange(self.max_iter):
            self._update_lambda(X)
            self._update_latent_factors(X)
            self._update_phi(X)
            bound = self._bound(X)
            improvement = (bound - old_bd) / abs(old_bd)
            if self.verbose:
                sys.stdout.write('\r\tAfter ITERATION: %d\tObjective: %.0f\t'
                                 'Old objective: %.0f\t'
                                 'Improvement: %.5f' % (i, bound, old_bd,
                                                        improvement))
                sys.stdout.flush()
            if np.abs(improvement) < self.tol:
                break
            old_bd = bound
        if self.verbose:
            sys.stdout.write('\n')
        pass
    def _update_lambda(self,X):
        self.gamma_lambda = self.a+self.phi_var_data.sum(axis=0,keepdims=True)
        self.rho_lambda = self.b
        for mode in range(X.shape[1]-1):
            self.rho_lambda+=self.Eb[mode].sum(axis=0,keepdims=True)
        self.Elambda,self.Eloglambda=_compute_expectations(self.gamma_lambda, self.rho_lambda)

        
    def _update_phi(self,X):
        self.phi_var = np.zeros((X.shape[0], self.n_components)) ### start zeroing everything
        for mode in range(X.shape[1]-1): ### for each element of the mode, minus the last one, which is the value
            # select the non-zero evidence elements of the latent factor of each mode
            # [m1, m2, m3, m4, m5, v] => Data[m1,m2,m3,m5]=v, so for each mode i we select m_i = {index non zero}
            self.phi_var = np.add(self.phi_var, np.exp(self.Elogb[mode][X[:,mode], :]))
        self.phi_var = np.add(self.phi_var, np.exp(self.Eloglambda)) 
        self.phi_var = np.divide(self.phi_var, np.sum(self.phi_var, axis=1)[:, np.newaxis])
        self.phi_var_data =X[:,-1,np.newaxis]*self.phi_var ### X[m1, m2, m3, m4, m5, v][-1]=v
        
    def _update_latent_factors(self, X):
        # variational parameters modes
        for mode in range(X.shape[1]-1):
            self.gamma_b[mode]=self.a + npi.group_by(X[:,mode]).sum(self.phi_var_data)[1]
            self.rho_b[mode]=self.b + self.Elambda*np.prod(self.partial_sums[:mode]+self.partial_sums[(mode+1):], axis=0)
            tempE, tempElog = _compute_expectations(self.gamma_b[mode], self.rho_b[mode])
            self.partial_sums[mode]=np.sum(tempE,axis=0,keepdims=True) ### shape=(1,self.n_components)
            self.Eb[mode]=tempE
            self.Elogb[mode]=tempElog

    def _xexplog(self):
        '''
        sum_k exp(E[log theta_{ik} * beta_{kd}])
        '''
        return np.dot(np.exp(self.Elogt), np.exp(self.Elogb))

    def _bound(self, X):
        bound=0
        bound+=np.sum(self.phi_var_data*self.Eloglambda)
        for mode in range(X.shape[1]-1):
            bound+=np.sum(self.phi_var_data*self.Elogb[mode][X[:,mode], :])
            bound-=np.sum(self.Elambda*np.prod(self.partial_sums,axis=0))
            bound += _gamma_term(self.a, self.a ,
                             self.gamma_b[mode], self.rho_b[mode],
                             self.Eb[mode], self.Elogb[mode])
        bound += _gamma_term(self.a, self.a ,
                         self.gamma_lambda, self.rho_lambda,
                         self.Elambda, self.Eloglambda)
        bound -= np.sum(np.log(self.phi_var+0.00000000000001)*self.phi_var_data)



        return bound


def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    #beta=beta.reshape((beta.shape[0], 1))
    # TODO: maybe use 1-dimensional beta and broadcast in the appropriate dimension... save memory if it is a problem
    return (alpha / beta, special.psi(alpha) - np.log(beta))


def _gamma_term(a, b, shape, rate, Ex, Elogx):
    return np.sum((a - shape) * Elogx - (b - rate) * Ex +
                  (special.gammaln(shape) - shape * np.log(rate)))


def _sum_product_newaxis1(auxvar, data, axis=1):
    return np.sum(auxvar * data[np.newaxis, :, :], axis=axis)

def _compute_entropy(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute Entropy[x]
    '''
    #beta=beta.reshape((beta.shape[0], 1))
    # TODO: use 1-dimensional beta and broadcast in the appropriate dimension... save memory if it is a problem
    return alpha+(1-alpha)*special.psi(alpha) - np.log(beta)+special.gammaln(alpha)