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

cdef class PCA:
  """Principal component analysis."""
  cdef openann.PCA *thisptr

  def __init__(self, n_components, whiten=True):
    self.thisptr = new openann.PCA(n_components, whiten)

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

  def explained_variance_ratio(self):
    cdef openann.VectorXd evr_eigen = self.thisptr.explainedVarianceRatio()
    return __vector_eigen_to_numpy__(&evr_eigen)

cdef class KMeans:
  """K-Means clustering."""
  cdef openann.KMeans *thisptr

  def __init__(self, n_inputs, n_centers):
    self.thisptr = new openann.KMeans(n_inputs, n_centers)

  def __dealloc__(self):
    del self.thisptr

  def update(self, X):
    cdef openann.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    self.thisptr.update(deref(X_eigen))

  def transform(self, X):
    cdef openann.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    cdef openann.MatrixXd Y_eigen = self.thisptr.transform(deref(X_eigen))
    return __matrix_eigen_to_numpy__(&Y_eigen)

  def get_centers(self):
    cdef openann.MatrixXd C_eigen = self.thisptr.getCenters()
    return __matrix_eigen_to_numpy__(&C_eigen)
