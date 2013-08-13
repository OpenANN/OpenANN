cdef class Normalization:
  """Normalize features to mean 0 and standard deviation 1."""
  cdef cbindings.Normalization *thisptr

  def __init__(self):
    self.thisptr = new cbindings.Normalization()

  def __dealloc__(self):
    del self.thisptr

  def fit(self, X):
    cdef cbindings.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    self.thisptr.fit(deref(X_eigen))
    return self

  def transform(self, X):
    cdef cbindings.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    cdef cbindings.MatrixXd Y_eigen = self.thisptr.transform(deref(X_eigen))
    return __matrix_eigen_to_numpy__(&Y_eigen)

  def get_mean(self):
    cdef cbindings.MatrixXd m_eigen = self.thisptr.getMean()
    return __matrix_eigen_to_numpy__(&m_eigen)

  def get_std(self):
    cdef cbindings.MatrixXd s_eigen = self.thisptr.getStd()
    return __matrix_eigen_to_numpy__(&s_eigen)

cdef class PCA:
  """Principal component analysis."""
  cdef cbindings.PCA *thisptr

  def __init__(self, n_components, whiten=True):
    self.thisptr = new cbindings.PCA(n_components, whiten)

  def __dealloc__(self):
    del self.thisptr

  def fit(self, X):
    cdef cbindings.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    self.thisptr.fit(deref(X_eigen))
    return self

  def transform(self, X):
    cdef cbindings.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    cdef cbindings.MatrixXd Y_eigen = self.thisptr.transform(deref(X_eigen))
    return __matrix_eigen_to_numpy__(&Y_eigen)

  def explained_variance_ratio(self):
    cdef cbindings.VectorXd evr_eigen = self.thisptr.explainedVarianceRatio()
    return __vector_eigen_to_numpy__(&evr_eigen)

cdef class ZCAWhitening:
  """Zero phase component analysis."""
  cdef cbindings.ZCAWhitening *thisptr

  def __init__(self):
    self.thisptr = new cbindings.ZCAWhitening()

  def __dealloc__(self):
    del self.thisptr

  def fit(self, X):
    cdef cbindings.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    self.thisptr.fit(deref(X_eigen))
    return self

  def transform(self, X):
    cdef cbindings.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    cdef cbindings.MatrixXd Y_eigen = self.thisptr.transform(deref(X_eigen))
    return __matrix_eigen_to_numpy__(&Y_eigen)
