class Activation:
  """Activation function."""
  LOGISTIC = openann.LOGISTIC
  TANH = openann.TANH
  TANH_SCALED = openann.TANH_SCALED
  RECTIFIER = openann.RECTIFIER
  LINEAR = openann.LINEAR

class Error:
  """Error function."""
  SSE = openann.SSE
  MSE = openann.MSE
  CE = openann.CE

cdef class Net:
  """A multilayer feedforward network."""
  cdef openann.Net *thisptr
  cdef object layers

  def __cinit__(self):
    self.thisptr = new openann.Net()
    self.layers = []

  def __dealloc__(self):
    del self.thisptr

  def input_layer(self, dim1, dim2=1, dim3=1):
    """Add an input layer."""
    self.thisptr.inputLayer(dim1, dim2, dim3)
    return self

  def alpha_beta_filter_layer(self, delta_t, std_dev=0.05):
    """Add an alpha-beta filter layer."""
    self.thisptr.alphaBetaFilterLayer(delta_t, std_dev)
    return self

  def fully_connected_layer(self, units, act, std_dev=0.05, bias=True):
    """Add a fully connected layer."""
    self.thisptr.fullyConnectedLayer(units, act, std_dev, bias)
    return self

  def compressed_layer(self, units, params, act, compression, std_dev=0.05, bias=True):
    """Add a compressed layer."""
    cdef char* comp = compression
    self.thisptr.compressedLayer(units, params, act, string(comp), std_dev, bias)
    return self

  def extreme_layer(self, units, act, std_dev=5.0, bias=True):
    """Add an extreme layer."""
    self.thisptr.extremeLayer(units, act, std_dev, bias)
    return self

  def convolutional_layer(self, featureMaps, kernelRows, kernelCols, act, std_dev=0.05, bias=True):
    """Add a convolutional layer."""
    self.thisptr.convolutionalLayer(featureMaps, kernelRows, kernelCols, act, std_dev, bias)
    return self

  def subsampling_layer(self, kernelRows, kernelCols, act, std_dev=0.05, bias=True):
    """Add a subsampling layer."""
    self.thisptr.subsamplingLayer(kernelRows, kernelCols, act, std_dev, bias)
    return self

  def maxpooling_layer(self, kernelRows, kernelCols):
    """Add a maxpooling layer."""
    self.thisptr.maxPoolingLayer(kernelRows, kernelCols)
    return self

  def local_response_normalization_layer(self, k, n, alpha, beta):
    """Add a local response normalization layer."""
    self.thisptr.localReponseNormalizationLayer(k, n, alpha, beta)
    return self

  def output_layer(self, units, act, std_dev=0.05):
    """Add an output layer."""
    self.thisptr.outputLayer(units, act, std_dev)
    return self

  def dropout_layer(self, dropout_probability):
    """Add a dropout layer."""
    self.thisptr.dropoutLayer(dropout_probability)
    return self

  def add_layer(self, layer):
    """Add a layer."""
    cdef int layers = self.thisptr.numberOflayers()
    self.thisptr.addLayer((<Layer?>layer).construct())
    self.layers.append(layer)

  def add_output_layer(self, layer):
    """Add an output layer."""
    cdef int layers = self.thisptr.numberOflayers()
    self.thisptr.addOutputLayer((<Layer?>layer).construct())
    self.layers.append(layer)

  def compressed_output_layer(self, units, params, act, compression, std_dev=0.05):
    """Add a compressed output layer."""
    cdef char* comp = compression
    self.thisptr.compressedOutputLayer(units, params, act, string(comp), std_dev)
    return self

  def set_regularization(self, l1_penalty=0.0, l2_penalty=0.0, max_squared_weight_norm=0.0):
    """Set regularization coefficients."""
    self.thisptr.setRegularization(l1_penalty, l2_penalty, max_squared_weight_norm)

  def set_error_function(self, err):
    """Set the error function."""
    self.thisptr.setErrorFunction(err)
    return self

  def use_dropout(self, activate):
    """(De)activate dropout."""
    self.thisptr.useDropout(activate)

  def predict(self, x_numpy):
    """Predict output for a given input."""
    cdef openann.VectorXd* x_eigen = __vector_numpy_to_eigen__(x_numpy)
    cdef openann.VectorXd y_eigen = self.thisptr.predict(deref(x_eigen))
    del x_eigen
    return __vector_eigen_to_numpy__(&y_eigen)

  cdef openann.OutputInfo last_output_info(self):
    cdef int layers = self.thisptr.numberOflayers()
    cdef openann.OutputInfo info = self.thisptr.getOutputInfo(layers - 1)
    return info

  def parameter_size(self):
    """Get number of parameters."""
    return self.thisptr.dimension()
