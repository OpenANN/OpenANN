cdef openann.VectorXd* __vector_numpy_to_eigen__(object x_numpy):
  cdef int dim_size = x_numpy.size
  cdef openann.VectorXd* x_eigen = new openann.VectorXd(dim_size)
  for r in range(x_numpy.size):
    x_eigen.data()[r] = x_numpy[r]
  return x_eigen

cdef object __vector_eigen_to_numpy__(openann.VectorXd* x_eigen):
  x_numpy = numpy.ndarray(shape=(x_eigen.rows(), 1))
  for r in range(x_eigen.rows()):
    x_numpy[r] = x_eigen.data()[r]
  return x_numpy


cdef openann.MatrixXd* __matrix_numpy_to_eigen__(object X_numpy):
  cdef openann.MatrixXd* X_eigen = new openann.MatrixXd(X_numpy.shape[0], X_numpy.shape[1])
  flatten_matrix = numpy.reshape(X_numpy, X_numpy.size, order='F')
  for r in range(X_numpy.size):
    X_eigen.data()[r] = flatten_matrix[r]
  return X_eigen

cdef object __matrix_eigen_to_numpy__(openann.MatrixXd* X_eigen):
  x_numpy = numpy.ndarray(shape=(X_eigen.rows(), X_eigen.cols()))
  for i in range(X_eigen.rows()):
    for j in range(X_eigen.cols()):
        x_numpy[i,j] = X_eigen.get(i, j)
  return x_numpy


