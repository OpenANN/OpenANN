cdef class Dataset:
  cdef openann.MatrixXd* input
  cdef openann.MatrixXd* output
  cdef openann.DirectStorageDataSet* dataset

  def __cinit__(self, input, output):
    self.input = __matrix_numpy_to_eigen__(input)
    self.output = __matrix_numpy_to_eigen__(output)
    self.dataset = new openann.DirectStorageDataSet(deref(self.input), deref(self.output), openann.NONE, 0)

  def __cinit__(self):
    self.input = NULL
    self.output = NULL
    self.dataset = NULL

  def __dealloc__(self):
    if(self.dataset != NULL):
      del self.dataset
    if(self.input != NULL):
      del self.input
    if(self.output != NULL):
      del self.output

  def samples(self):
    if self.dataset != NULL:
      return self.dataset.samples()
    else:
      return 0

  def inputs(self):
    if self.dataset != NULL:
      return self.dataset.inputs()
    else:
      return 0

  def outputs(self):
    if self.dataset != NULL:
      return self.dataset.outputs()
    else:
      return 0

  def instance(self, i):
    if self.dataset != NULL:
      return __vector_eigen_to_numpy__(&self.dataset.getInstance(i))
    else:
      return 0

  def target(self, i):
    if self.dataset != NULL:
      return __vector_eigen_to_numpy__(&self.dataset.getTarget(i))
    else:
      return 0


def load_from_libsvm(filename):
  cdef openann.MatrixXd* input = new openann.MatrixXd()
  cdef openann.MatrixXd* output = new openann.MatrixXd()
  dset = Dataset()
  dset.input = input
  dset.output = output
  openann.load_from_libsvm(deref(input), deref(output), filename, 0)
  dset.dataset = new openann.DirectStorageDataSet(deref(input), deref(output), openann.NONE, 0)
  return dset 

