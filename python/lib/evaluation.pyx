
def sse(learner, dataset):
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  return openann.sse(deref(net), deref(ds))

def mse(learner, dataset):
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  return openann.mse(deref(net), deref(ds))

def rmse(learner, dataset):
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  return openann.rmse(deref(net), deref(ds))

def accuracy(learner, dataset):
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  return openann.accuracy(deref(net), deref(ds))

def classification_hits(learner, dataset):
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  return openann.classificationHits(deref(net), deref(ds))

def cross_validation(folds, learner, dataset, optimizer):
  cdef openann.Learner *net = (<Net?>learner).thisptr
  cdef openann.DataSet *ds = (<Dataset?>dataset).storage
  cdef openann.Optimizer *opt = (<Optimizer?>optimizer).thisptr
  openann.crossValidation(folds, deref(net), deref(ds), deref(opt))


