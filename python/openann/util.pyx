@cython.boundscheck(False)
cdef cbindings.VectorXd* __vector_numpy_to_eigen__(numpy.ndarray x_numpy):
  assert x_numpy.ndim == 1, \
      "Vector must have exactly one dimension instead of %d" % x_numpy.ndim
  cdef int dim_size = x_numpy.size
  cdef cbindings.VectorXd* x_eigen = new cbindings.VectorXd(dim_size)
  cdef int rows = x_numpy.shape[0]
  for r in range(rows):
    x_eigen.data()[r] = x_numpy[r]
  return x_eigen

@cython.boundscheck(False)
cdef numpy.ndarray __vector_eigen_to_numpy__(cbindings.VectorXd* x_eigen):
  cdef numpy.ndarray[numpy.float64_t, ndim=1] x_numpy = \
      numpy.ndarray(shape=(x_eigen.rows(), 1))
  for r in range(x_eigen.rows()):
    x_numpy[r] = x_eigen.data()[r]
  return x_numpy

@cython.boundscheck(False)
cdef cbindings.MatrixXd* __matrix_numpy_to_eigen__(numpy.ndarray X_numpy):
  assert X_numpy.ndim == 2, \
      "Matrix must have exactly two dimensions instead of %d" % X_numpy.ndim
  cdef cbindings.MatrixXd* X_eigen = new cbindings.MatrixXd(X_numpy.shape[0],
                                                            X_numpy.shape[1])
  cdef int rows = X_numpy.shape[0]
  cdef int cols = X_numpy.shape[1]
  for r in range(rows):
    for c in range(cols):
      X_eigen.data()[rows * c + r] = X_numpy[r, c]
  return X_eigen

@cython.boundscheck(False)
cdef numpy.ndarray __matrix_eigen_to_numpy__(cbindings.MatrixXd* X_eigen):
  cdef numpy.ndarray[numpy.float64_t, ndim=2] x_numpy = \
      numpy.ndarray(shape=(X_eigen.rows(), X_eigen.cols()))
  for r in range(X_eigen.rows()):
    for c in range(X_eigen.cols()):
        x_numpy[r, c] = X_eigen.coeff(r, c)
  return x_numpy

@cython.boundscheck(False)
cdef numpy.ndarray __matrix_eigen_to_numpy_int__(cbindings.MatrixXi* X_eigen):
  cdef numpy.ndarray[numpy.int_t, ndim=2] x_numpy = \
      numpy.ndarray(shape=(X_eigen.rows(), X_eigen.cols()))
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
  def set_disabled(object cls):
    cbindings.setDisabled()

  @classmethod
  def set_error(object cls):
    cbindings.setError()

  @classmethod
  def set_info(object cls):
    cbindings.setInfo()

  @classmethod
  def set_debug(object cls):
    cbindings.setDebug()

  @classmethod
  def debug(object cls, char* text):
    cbindings.write(cbindings.Log().get(Log.DEBUG, ""), <char*?>text)

  @classmethod
  def info(object cls,  char* text):
    cbindings.write(cbindings.Log().get(Log.INFO, ""), <char*?>text)

  @classmethod
  def error(object cls,  char* text):
    cbindings.write(cbindings.Log().get(Log.ERROR, ""), <char*?>text)


cdef class RandomNumberGenerator:
  """Controls random number generation in OpenANN."""
  cdef cbindings.RandomNumberGenerator *thisptr

  def __cinit__(object self):
    self.thisptr = new cbindings.RandomNumberGenerator()

  def __dealloc__(object self):
    del self.thisptr

  def seed(object self, int s):
    self.thisptr.seed(s)


cdef class OpenANN:
  """OpenANN library infos."""
  @classmethod
  def version(object cls):
    return cbindings.VERSION

  @classmethod
  def url(object cls):
    return cbindings.URL

  @classmethod
  def description(object cls):
    return cbindings.DESCRIPTION

  @classmethod
  def compilation_time(object cls):
    return cbindings.COMPILATION_TIME

  @classmethod
  def compilation_flags(object cls):
    return cbindings.COMPILATION_FLAGS


def _use_all_cores():
  cbindings.useAllCores()


_use_all_cores()
