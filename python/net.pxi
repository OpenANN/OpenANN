cdef class Learner:
  """Base class of learning algorithms."""
  cdef cbindings.Learner *learner

class Activation:
  """Activation function."""
  LOGISTIC = cbindings.LOGISTIC
  TANH = cbindings.TANH
  TANH_SCALED = cbindings.TANH_SCALED
  RECTIFIER = cbindings.RECTIFIER
  LINEAR = cbindings.LINEAR

class Error:
  """Error function."""
  MSE = cbindings.MSE
  CE = cbindings.CE

cdef class Net(Learner):
  """A multilayer feedforward network."""
  cdef cbindings.Net *thisptr

  def __cinit__(self):
    self.thisptr = new cbindings.Net()
    self.learner = self.thisptr

  def __dealloc__(self):
    del self.thisptr

  def initialize(self):
    self.thisptr.initialize()

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

  def restricted_boltzmann_machine_layer(self, units, cd_n=1, std_dev=0.01,
                                         backprop=True):
    """Add an RBM."""
    self.thisptr.restrictedBoltzmannMachineLayer(units, cd_n, std_dev,
                                                 backprop)

  def compressed_layer(self, units, params, act, compression, std_dev=0.05,
                       bias=True):
    """Add a compressed layer."""
    cdef char* comp = compression
    self.thisptr.compressedLayer(units, params, act, string(comp), std_dev,
                                 bias)
    return self

  def extreme_layer(self, units, act, std_dev=5.0, bias=True):
    """Add an extreme layer."""
    self.thisptr.extremeLayer(units, act, std_dev, bias)
    return self

  def intrinsic_plasticity_layer(self, target_mean, std_dev=1.0):
    """Add an intrinsic plasticity layer."""
    self.thisptr.intrinsicPlasticityLayer(target_mean, std_dev)

  def convolutional_layer(self, featureMaps, kernelRows, kernelCols, act,
                          std_dev=0.05, bias=True):
    """Add a convolutional layer."""
    self.thisptr.convolutionalLayer(featureMaps, kernelRows, kernelCols, act,
                                    std_dev, bias)
    return self

  def subsampling_layer(self, kernelRows, kernelCols, act, std_dev=0.05,
                        bias=True):
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

  def dropout_layer(self, dropout_probability):
    """Add a dropout layer."""
    self.thisptr.dropoutLayer(dropout_probability)
    return self

  def output_layer(self, units, act, std_dev=0.05):
    """Add an output layer."""
    self.thisptr.outputLayer(units, act, std_dev)
    return self

  def compressed_output_layer(self, units, params, act, compression,
                              std_dev=0.05):
    """Add a compressed output layer."""
    cdef char* comp = compression
    self.thisptr.compressedOutputLayer(units, params, act, string(comp),
                                       std_dev)
    return self

  def add_layer(self, layer):
    """Add a layer."""
    self.thisptr.addLayer((<Layer?>layer).construct())

  def add_output_layer(self, layer):
    """Add an output layer."""
    self.thisptr.addOutputLayer((<Layer?>layer).construct())

  def set_regularization(self, l1_penalty=0.0, l2_penalty=0.0,
                         max_squared_weight_norm=0.0):
    """Set regularization coefficients."""
    self.thisptr.setRegularization(l1_penalty, l2_penalty,
                                   max_squared_weight_norm)

  def set_error_function(self, err):
    """Set the error function."""
    self.thisptr.setErrorFunction(err)
    return self

  def use_dropout(self, activate):
    """(De)activate dropout."""
    self.thisptr.useDropout(activate)

  def predict(self, x_numpy):
    """Predict output for given inputs, each row represents an instance."""
    x_numpy = numpy.atleast_2d(x_numpy)
    cdef cbindings.MatrixXd* x_eigen = __matrix_numpy_to_eigen__(x_numpy)
    cdef cbindings.MatrixXd y_eigen = self.thisptr.predict(deref(x_eigen))
    del x_eigen
    return __matrix_eigen_to_numpy__(&y_eigen)

  cdef number_of_layers(self):
    """Get number of layers."""
    return self.thisptr.numberOflayers()

  def get_layer(self, l):
    """Get the l-th layer."""
    cdef cbindings.Layer* layer = &self.thisptr.getLayer(l)
    layer_object = Layer()
    layer_object.thisptr = layer
    return layer_object

  cdef cbindings.OutputInfo output_info(self, layer):
    cdef cbindings.OutputInfo info = self.thisptr.getOutputInfo(layer)
    return info

  def dimension(self):
    """Get number of parameters."""
    return self.thisptr.dimension()

  def set_parameters(self, parameters):
    """Set parameters of the network."""
    cdef cbindings.VectorXd* params_eigen = __vector_numpy_to_eigen__(parameters)
    self.thisptr.setParameters(deref(params_eigen))

  def current_parameters(self):
    """Get current parameters."""
    cdef cbindings.VectorXd params_eigen = self.thisptr.currentParameters()
    return __vector_eigen_to_numpy__(&params_eigen)

  def save(self, file_name):
    cdef char* fn = file_name
    self.thisptr.save(string(fn))

  def load(self, file_name):
    cdef char* fn = file_name
    self.thisptr.load(string(fn))


cdef class RBM(Learner):
  """Restricted Boltzmann machine."""
  cdef cbindings.RBM *thisptr

  def __init__(self, D, H, cd_N=1, std_dev=0.01, backprop=True, l1penalty=0.0,
               l2penalty=0.0):
    cdef cbindings.Regularization* regularization = \
        new cbindings.Regularization(l1penalty, l2penalty, 0.0)
    self.thisptr = new cbindings.RBM(D, H, cd_N, std_dev, backprop,
                                     deref(regularization))
    del regularization
    self.learner = self.thisptr

  def __dealloc__(self):
    del self.thisptr

  def predict(self, x_numpy):
    x_numpy = numpy.atleast_2d(x_numpy)
    cdef cbindings.MatrixXd* x_eigen = __matrix_numpy_to_eigen__(x_numpy)
    cdef cbindings.MatrixXd y_eigen = self.thisptr.predict(deref(x_eigen))
    del x_eigen
    return __matrix_eigen_to_numpy__(&y_eigen)

  def visible_units(self):
    return self.thisptr.visibleUnits()

  def hidden_units(self):
    return self.thisptr.hiddenUnits()

  def get_weights(self):
    cdef cbindings.MatrixXd weights = self.thisptr.getWeights()
    return __matrix_eigen_to_numpy__(&weights)

  def get_visible_probs(self):
    cdef cbindings.MatrixXd probs = self.thisptr.getVisibleProbs()
    return __matrix_eigen_to_numpy__(&probs)

  def get_visible_sample(self):
    cdef cbindings.MatrixXd sample = self.thisptr.getVisibleSample()
    return __matrix_eigen_to_numpy__(&sample)

  def reconstruct_prob(self, n, steps):
    cdef cbindings.MatrixXd prop = self.thisptr.reconstructProb(n, steps)
    return __matrix_eigen_to_numpy__(&prop)

  def sample_H_given_V(self):
    self.thisptr.sampleHgivenV()

  def sample_V_given_H(self):
    self.thisptr.sampleVgivenH()


cdef class SparseAutoEncoder(Learner):
  """Sparse auto-encoder."""
  cdef cbindings.SparseAutoEncoder *thisptr

  def __init__(self, D, H, beta, rho, lmbda, act):
    self.thisptr = new cbindings.SparseAutoEncoder(D, H, beta, rho, lmbda, act)
    self.learner = self.thisptr

  def __dealloc__(self):
    del self.thisptr

  def predict(self, x_numpy):
    x_numpy = numpy.atleast_2d(x_numpy)
    cdef cbindings.MatrixXd* x_eigen = __matrix_numpy_to_eigen__(x_numpy)
    cdef cbindings.MatrixXd y_eigen = self.thisptr.predict(deref(x_eigen))
    del x_eigen
    return __matrix_eigen_to_numpy__(&y_eigen)

  def get_input_weights(self):
    """Get weight matrix between input and hidden layer."""
    cdef cbindings.MatrixXd weights = self.thisptr.getInputWeights()
    return __matrix_eigen_to_numpy__(&weights)

  def get_output_weights(self):
    """Get weight matrix between hidden and output layer."""
    cdef cbindings.MatrixXd weights = self.thisptr.getOutputWeights()
    return __matrix_eigen_to_numpy__(&weights)

  def reconstruct(self, x):
    """Reconstruct input."""
    cdef cbindings.VectorXd* x_eigen = __vector_numpy_to_eigen__(x)
    cdef cbindings.VectorXd recon = self.thisptr.reconstruct(deref(x_eigen))
    return __vector_eigen_to_numpy__(&recon)
