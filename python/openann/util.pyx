cdef cbindings.VectorXd* __vector_numpy_to_eigen__(object x_numpy):
  assert len(x_numpy.shape) == 1, \
      "Vector must have exactly one dimension instead of %d" % \
      len(x_numpy.shape)
  cdef int dim_size = x_numpy.size
  cdef cbindings.VectorXd* x_eigen = new cbindings.VectorXd(dim_size)
  for r in range(x_numpy.size):
    x_eigen.data()[r] = x_numpy[r]
  return x_eigen

cdef object __vector_eigen_to_numpy__(cbindings.VectorXd* x_eigen):
  x_numpy = numpy.ndarray(shape=(x_eigen.rows(), 1))
  for r in range(x_eigen.rows()):
    x_numpy[r] = x_eigen.data()[r]
  return x_numpy

cdef cbindings.MatrixXd* __matrix_numpy_to_eigen__(object X_numpy):
  assert len(X_numpy.shape) == 2, \
      "Matrix must have exactly two dimensions instead of %d" % \
      len(X_numpy.shape)
  cdef cbindings.MatrixXd* X_eigen = new cbindings.MatrixXd(X_numpy.shape[0],
                                                            X_numpy.shape[1])
  for r in range(X_numpy.shape[0]):
    for c in range(X_numpy.shape[1]):
      X_eigen.data()[X_numpy.shape[0] * c + r] = X_numpy[r, c]
  return X_eigen

cdef object __matrix_eigen_to_numpy__(cbindings.MatrixXd* X_eigen):
  x_numpy = numpy.ndarray(shape=(X_eigen.rows(), X_eigen.cols()))
  for r in range(X_eigen.rows()):
    for c in range(X_eigen.cols()):
        x_numpy[r, c] = X_eigen.coeff(r, c)
  return x_numpy

cdef object __matrix_eigen_to_numpy_int__(cbindings.MatrixXi* X_eigen):
  x_numpy = numpy.ndarray(shape=(X_eigen.rows(), X_eigen.cols()))
  for r in range(X_eigen.rows()):
    for c in range(X_eigen.cols()):
        x_numpy[r, c] = X_eigen.coeff(r, c)
  return x_numpy


cdef class Log:
  """Provides logging functionality."""
  DISABLED = cbindings.DISABLED
  ERROR = cbindings.ERROR
  INFO = cbindings.INFO
  DEBUG = cbindings.DEBUG

  @classmethod
  def set_disabled(cls):
    cbindings.setDisabled()

  @classmethod
  def set_error(cls):
    cbindings.setError()

  @classmethod
  def set_info(cls):
    cbindings.setInfo()

  @classmethod
  def set_debug(cls):
    cbindings.setDebug()

  @classmethod
  def debug(cls, text):
    cbindings.write(cbindings.Log().get(Log.DEBUG, ""), <char*?>text)

  @classmethod
  def info(cls, text):
    cbindings.write(cbindings.Log().get(Log.INFO, ""), <char*?>text)

  @classmethod
  def error(cls, text):
    cbindings.write(cbindings.Log().get(Log.ERROR, ""), <char*?>text)


cdef class RandomNumberGenerator:
  """Controls random number generation in OpenANN."""
  cdef cbindings.RandomNumberGenerator *thisptr

  def __cinit__(self):
    self.thisptr = new cbindings.RandomNumberGenerator()

  def __dealloc__(self):
    del self.thisptr

  def seed(self, s):
    self.thisptr.seed(s)


cdef class OpenANN:
  """OpenANN library infos."""
  @classmethod
  def version(cls):
    return cbindings.VERSION

  @classmethod
  def url(cls):
    return cbindings.URL

  @classmethod
  def description(cls):
    return cbindings.DESCRIPTION

  @classmethod
  def compilation_time(cls):
    return cbindings.COMPILATION_TIME

  @classmethod
  def compilation_flags(cls):
    return cbindings.COMPILATION_FLAGS


def _use_all_cores():
  cbindings.useAllCores()


_use_all_cores()
