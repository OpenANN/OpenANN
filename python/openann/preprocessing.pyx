cdef class Normalization:
  """Normalize features to mean 0 and standard deviation 1."""
  cdef openann.Normalization *thisptr

  def __init__(self):
    self.thisptr = new openann.Normalization()

  def __dealloc__(self):
    del self.thisptr

  def fit(self, X):
    cdef openann.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    self.thisptr.fit(deref(X_eigen))
    return self

  def transform(self, X):
    cdef openann.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    cdef openann.MatrixXd Y_eigen = self.thisptr.transform(deref(X_eigen))
    return __matrix_eigen_to_numpy__(&Y_eigen)

  def get_mean(self):
    cdef openann.MatrixXd m_eigen = self.thisptr.getMean()
    return __matrix_eigen_to_numpy__(&m_eigen)

  def get_std(self):
    cdef openann.MatrixXd s_eigen = self.thisptr.getStd()
    return __matrix_eigen_to_numpy__(&s_eigen)
