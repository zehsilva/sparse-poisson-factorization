# sparse-poisson-factorization 
Non-negative probabilistic Poisson-gamma matrix factorization and tensor CP decomposition with sparse input and internal components. 
This is a (more) memory efficient version of Poisson Factorization in pure python that takes advantage of the fact that inference updates for Poisson Factorization needs to be performed only on the non-zero entries of the input matrix/tensor.
Both models consist on a hierarchical probabilistic model with a Poisson likelihood for the matrix entries, and Gamma distributed latent factors vectors for rows and columns (in the matrix case) or each mode (in the tensor case, effectively meaning that we are performing a non-negative CP decomposition).    

For the matrix case the expected value of latent factor can be accessed via the attributes `Eb` and `Et` of the poisson factorization object, while for the tensor CP decomposition the expected value of latent factors can be acessed via the attribute `Eb` indexed by the mode. 
 
## Dependencies:
- numpy_indexed
- scipy
- numpy
- scikit-learn

## Usage

The input for the train method on the matrix factorization should be a numpy array representing a sparse matrix. An array with shape (N,3), where N is the number of non-zero entries and for each non-zero entry an array with `[row, column, value]` ).
A dense array equivalent for each entry would be `Matrix[row,colum]=value`.

Example:

`[ [row_1, col_1, matrix_entry_row_1_col_1],`

 ` [row_2, col_2, matrix_entry_row_2_col_2],`

 ` [row_3, col_3, matrix_entry_row_3_col_3],`

 ` ... ]`


The input for the train method on the tensor factorization should be a numpy array representing a sparse vector. An array of shape (N,N_modes+1), where N is the number of non-zero entries, N_mode is the number of modes in the tensor, and for each non-zero entry an array of size N_modes+1 with `[mode1, mode2, ... , mode_N,value]`.
A dense tensor array equivalent for each entry would be  `Tensor[mode1, mode2, ... , mode_N]=value`.

Example:

`[ [mode1_1, mode2_1, ... , mode_N_1, value_1],`

`  [mode1_2, mode2_2, ... , mode_N_2, value_2],`

`  ... ]`


To use sparse poisson matrix factorization add the following import:


    import sparse_poisson as sp

    poisson = sp.PoissonMF(n_components=15,max_iter=1000,smoothness=0.1,verbose=True,tol=0.0001,a=0.1,b=0.1)
    poisson.fit(X)`

To use sparse poisson tensor factorization add the following import:


    import sparse_tensor as st

    poisson = st.PoissonTF(n_components=15,max_iter=1000,smoothness=0.1,verbose=True,tol=0.0001,a=0.1,b=0.1)
    poisson.fit(X)`


 This implementation is modification of the code found in [https://github.com/dawenl/stochastic_PMF/blob/master/code/pmf.py] by Dawen Liang <dliang@ee.columbia.edu>

To understand the models implemented look at the Following bibliography
    - [https://arxiv.org/abs/1311.1704](Scalable Recommendation with Poisson Factorization)
    - [https://arxiv.org/abs/1506.03493](Bayesian Poisson Tensor Factorization for Inferring Multilateral Relations from Sparse Dyadic Event Counts)
