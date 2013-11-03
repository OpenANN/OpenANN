cdef class DataSet:
  """Contains instances and targets."""
  cdef cbindings.MatrixXd* input
  cdef cbindings.MatrixXd* output
  cdef cbindings.DataSet* storage

  def __cinit__(self, input=None, output=None):
    if input != None:
      self.input = __matrix_numpy_to_eigen__(input)
    else:
      self.input = NULL
    if output != None:
      self.output = __matrix_numpy_to_eigen__(output)
    else:
      self.output = NULL
    if self.input != NULL:
      self.storage = new cbindings.DirectStorageDataSet(self.input, self.output)
    else:
      self.storage = NULL

  def __dealloc__(self):
    if(self.storage != NULL):
      del self.storage
    if(self.input != NULL):
      del self.input
    if(self.output != NULL):
      del self.output

  def samples(self):
    if self.storage != NULL:
      return self.storage.samples()
    else:
      return 0

  def inputs(self):
    if self.storage != NULL:
      return self.storage.inputs()
    else:
      return 0

  def outputs(self):
    if self.storage != NULL:
      return self.storage.outputs()
    else:
      return 0

  def instance(self, i):
    if self.storage != NULL:
      return __vector_eigen_to_numpy__(&self.storage.getInstance(i))
    else:
      return 0

  def target(self, i):
    if self.storage != NULL:
      return __vector_eigen_to_numpy__(&self.storage.getTarget(i))
    else:
      return 0


def load_from_libsvm(filename):
  """Load dataset from libsvm file."""
  cdef cbindings.MatrixXd* input = new cbindings.MatrixXd()
  cdef cbindings.MatrixXd* output = new cbindings.MatrixXd()
  ds = DataSet()
  ds.input = input
  ds.output = output
  cbindings.libsvm_load(deref(input), deref(output), filename, 0)
  ds.storage = new cbindings.DirectStorageDataSet(input, output)
  return ds 


cdef class DataStream:
  """Streams training data for online training."""
  cdef cbindings.DataStream* thisptr

  def __cinit__(self, cache_size):
    self.thisptr = new cbindings.DataStream(cache_size)

  def __dealloc__(self):
    del self.thisptr

  def set_learner(self, learner):
    self.thisptr.setLearner(deref((<Learner?>learner).learner))
    return self

  def set_optimizer(self, optimizer):
    self.thisptr.setOptimizer(deref((<Optimizer?>optimizer).thisptr))
    return self

  def add_sample(self, x, y=None):
    cdef cbindings.VectorXd* x_eigen = __vector_numpy_to_eigen__(x)
    cdef cbindings.VectorXd* y_eigen = NULL
    if y is not None:
      y_eigen = __vector_numpy_to_eigen__(y)
    self.thisptr.addSample(x_eigen, y_eigen)
    del x_eigen
    if y is not None:
      del y_eigen
