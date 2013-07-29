def sse(learner, dataset):
  """Compute sum of squared errors."""
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<Dataset?>dataset).storage
  return cbindings.sse(deref(net), deref(ds))

def mse(learner, dataset):
  """Compute mean squared error."""
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<Dataset?>dataset).storage
  return cbindings.mse(deref(net), deref(ds))

def rmse(learner, dataset):
  """Compute mean squared error."""
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<Dataset?>dataset).storage
  return cbindings.rmse(deref(net), deref(ds))

def accuracy(learner, dataset):
  """Compute classification accuracy."""
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<Dataset?>dataset).storage
  return cbindings.accuracy(deref(net), deref(ds))

def confusion_matrix(learner, dataset):
  """Compute confusion matrix."""
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<Dataset?>dataset).storage
  cdef cbindings.MatrixXi conf_mat = cbindings.confusionMatrix(deref(net),
                                                               deref(ds))
  return __matrix_eigen_to_numpy_int__(&conf_mat)

def classification_hits(learner, dataset):
  """Compute number of correct predictions."""
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<Dataset?>dataset).storage
  return cbindings.classificationHits(deref(net), deref(ds))

def cross_validation(folds, learner, dataset, optimizer):
  """Perform cross validation."""
  cdef cbindings.Learner *net = (<Net?>learner).thisptr
  cdef cbindings.DataSet *ds = (<Dataset?>dataset).storage
  cdef cbindings.Optimizer *opt = (<Optimizer?>optimizer).thisptr
  return cbindings.crossValidation(folds, deref(net), deref(ds), deref(opt))
