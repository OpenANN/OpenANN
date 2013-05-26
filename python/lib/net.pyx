class Activation:
  LOGISTIC = openann.LOGISTIC
  TANH = openann.TANH
  TANH_SCALED = openann.TANH_SCALED
  RECTIFIER = openann.RECTIFIER
  LINEAR = openann.LINEAR

class Error:
  SSE = openann.SSE
  MSE = openann.MSE
  CE = openann.CE

cdef class Net:
  cdef openann.Net *thisptr
  cdef object layers

  def __cinit__(self):
    self.thisptr = new openann.Net()
    self.layers = []

  def __dealloc__(self):
    del self.thisptr

  def parameter_size(self):
    return self.thisptr.dimension()

  def input_layer(self, width, height, dim=1):
    self.thisptr.inputLayer(width, height, dim)
    return self

  def alpha_beta_filter_layer(self, delta_t, std_dev=0.05):
    self.thisptr.alphaBetaFilterLayer(delta_t, std_dev)
    return self

  def fully_connected_layer(self, units, act, std_dev=0.05, bias=True):
    self.thisptr.fullyConnectedLayer(units, act, std_dev, bias)
    return self

  def compressed_layer(self, units, params, act, compression, std_dev=0.05, bias=True):
    cdef char* comp = compression
    self.thisptr.compressedLayer(units, params, act, string(comp), std_dev, bias)
    return self

  def extreme_layer(self, units, act, std_dev=5.0, bias=True):
    self.thisptr.extremeLayer(units, act, std_dev, bias)
    return self

  def convolutional_layer(self, featureMaps, kernelRows, kernelCols, act, std_dev=0.05, bias=True):
    self.thisptr.convolutionalLayer(featureMaps, kernelRows, kernelCols, act, std_dev, bias)
    return self

  def subsampling_layer(self, kernelRows, kernelCols, act, std_dev=0.05, bias=True):
    self.thisptr.subsamplingLayer(kernelRows, kernelCols, act, std_dev, bias)
    return self

  def maxpooling_layer(self, kernelRows, kernelCols):
    self.thisptr.maxPoolingLayer(kernelRows, kernelCols)
    return self

  def local_response_normalization_layer(self, k, n, alpha, beta):
    self.thisptr.localReponseNormalizationLayer(k, n, alpha, beta)
    return self

  def output_layer(self, units, act, std_dev=0.05):
    self.thisptr.outputLayer(units, act, std_dev)
    return self

  def dropout_layer(self, dropout_probability):
    self.thisptr.dropoutLayer(dropout_probability)
    return self

  def add_layer(self, layer):
    cdef int layers = self.thisptr.numberOflayers()
    self.thisptr.addLayer((<Layer?>layer).construct())
    self.layers.append(layer)

  def add_output_layer(self, layer):
    cdef int layers = self.thisptr.numberOflayers()
    self.thisptr.addOutputLayer((<Layer?>layer).construct())
    self.layers.append(layer)

  def compressed_output_layer(self, units, params, act, compression, std_dev=0.05):
    cdef char* comp = compression
    self.thisptr.compressedOutputLayer(units, params, act, string(comp), std_dev)
    return self

  def set_error_function(self, err):
    self.thisptr.setErrorFunction(err)
    return self

  def use_dropout(self, activate):
    self.thisptr.useDropout(activate)

  def predict(self, x_numpy):
    cdef openann.VectorXd* x_eigen = __vector_numpy_to_eigen__(x_numpy)
    cdef openann.VectorXd y_eigen = self.thisptr.predict(deref(x_eigen))
    del x_eigen
    return __vector_eigen_to_numpy__(&y_eigen)

  cdef openann.OutputInfo last_output_info(self):
    cdef int layers = self.thisptr.numberOflayers()
    cdef openann.OutputInfo info = self.thisptr.getOutputInfo(layers - 1)
    return info


