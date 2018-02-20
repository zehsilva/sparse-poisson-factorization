# sparse-poisson-factorization
Poisson Matrix and Tensor Factorization with sparse input and internal representations.

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