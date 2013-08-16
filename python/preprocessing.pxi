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

cdef class KMeans:
  """K-Means clustering."""
  cdef cbindings.KMeans *thisptr

  def __init__(self, n_inputs, n_centers):
    self.thisptr = new cbindings.KMeans(n_inputs, n_centers)

  def fit(self, X):
    cdef cbindings.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    self.thisptr.fit(deref(X_eigen))
    return self

  def fit_partial(self, X):
    cdef cbindings.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    self.thisptr.fitPartial(deref(X_eigen))
    return self

  def transform(self, X):
    cdef cbindings.MatrixXd* X_eigen = __matrix_numpy_to_eigen__(X)
    cdef cbindings.MatrixXd Y_eigen = self.thisptr.transform(deref(X_eigen))
    return __matrix_eigen_to_numpy__(&Y_eigen)

  def get_centers(self):
    cdef cbindings.MatrixXd C_eigen = self.thisptr.getCenters()
    return __matrix_eigen_to_numpy__(&C_eigen)

def sample_random_patches(images, channels, rows, cols, samples, patch_rows,
                          patch_cols):
  cdef cbindings.MatrixXd* images_eigen = __matrix_numpy_to_eigen__(images)
  cdef cbindings.MatrixXd patches = cbindings.sampleRandomPatches(
      deref(images_eigen), channels, rows, cols, samples, patch_rows,
      patch_cols)
  return __matrix_eigen_to_numpy__(&patches)
