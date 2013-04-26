cdef class Dataset:
  cdef openann.MatrixXd* input
  cdef openann.MatrixXd* output
  cdef openann.DataSet* storage

  def __cinit__(self, input=None, output=None):
    if input != None and output != None:
      self.input = __matrix_numpy_to_eigen__(input)
      self.output = __matrix_numpy_to_eigen__(output)
      self.storage = new openann.DirectStorageDataSet(deref(self.input), deref(self.output))
    else:
      self.input = NULL
      self.output = NULL
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
  cdef openann.MatrixXd* input = new openann.MatrixXd()
  cdef openann.MatrixXd* output = new openann.MatrixXd()
  dset = Dataset()
  dset.input = input
  dset.output = output
  openann.libsvm_load(deref(input), deref(output), filename, 0)
  dset.storage = new openann.DirectStorageDataSet(deref(input), deref(output))
  return dset 

