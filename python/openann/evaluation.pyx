def sse(learner, dataset):
  """Compute sum of squared errors."""
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  return openann.sse(deref(net), deref(ds))

def mse(learner, dataset):
  """Compute mean squared error."""
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  return openann.mse(deref(net), deref(ds))

def rmse(learner, dataset):
  """Compute mean squared error."""
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  return openann.rmse(deref(net), deref(ds))

def accuracy(learner, dataset):
  """Compute classification accuracy."""
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  return openann.accuracy(deref(net), deref(ds))

def confusion_matrix(learner, dataset):
  """Compute confusion matrix."""
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  cdef openann.MatrixXi conf_mat = openann.confusionMatrix(deref(net),
                                                           deref(ds))
  return __matrix_eigen_to_numpy_int__(&conf_mat)

def classification_hits(learner, dataset):
  """Compute number of correct predictions."""
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  return openann.classificationHits(deref(net), deref(ds))

def cross_validation(folds, learner, dataset, optimizer):
  """Perform cross validation."""
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  cdef openann.Optimizer *opt = (<Optimizer?>optimizer).thisptr
  return openann.crossValidation(folds, deref(net), deref(ds), deref(opt))
