from libcpp cimport bool
from libcpp.string cimport string
from cython.operator cimport dereference as deref
import numpy

cdef extern from "OpenANN/ActivationFunctions.h" namespace "OpenANN":
  cdef enum c_ActivationFunction "OpenANN::ActivationFunction":
    LOGISTIC
    TANH
    TANH_SCALED
    RECTIFIER
    LINEAR

cdef extern from "OpenANN/optimization/StoppingCriteria.h" namespace "OpenANN":
  ctypedef struct c_StoppingCriteria "OpenANN::StoppingCriteria":
    int maximalFunctionEvaluations
    int maximalIterations
    int maximalRestarts
    double minimalValue
    double minimalValueDifferences
    double minimalSearchSpaceStep
  c_StoppingCriteria *new_StoppingCriteria "new OpenANN::StoppingCriteria" ()
  void del_StoppingCriteria "delete" (c_StoppingCriteria *ptr)

  ctypedef struct c_VectorXd "Eigen::VectorXd":
    double* data()
    int rows()
    double& get "operator()"(int rows)
  c_VectorXd *new_VectorXd "new Eigen::VectorXd" (c_VectorXd& vec)
  c_VectorXd *new_VectorXd "new Eigen::VectorXd" (int rows, int cols)
  void del_VectorXd "delete" (c_VectorXd *ptr)

  ctypedef struct c_MatrixXd "Eigen::MatrixXd":
    double& coeff(int row, int col)
    double* data()
    int rows()
    int cols()
    double& get "operator()"(int rows, int cols)
  c_MatrixXd *new_MatrixXd "new Eigen::MatrixXd" (int rows, int cols)
  void del_MatrixXd "delete" (c_MatrixXd *ptr)

cdef extern from "OpenANN/Learner.h" namespace "OpenANN":
  ctypedef struct c_Learner "OpenANN::Learner"

cdef extern from "OpenANN/Net.h" namespace "OpenANN":
  cdef enum c_ErrorFunction "OpenANN::ErrorFunction":
    NO_E_DEFINED
    SSE
    MSE
    CE

  ctypedef struct c_Net "OpenANN::Net":
    c_Net& inputLayer(int dim1, int dim2, int dim3, bool bias)
    c_Net& alphaBetaFilterLayer(double deltaT, double stdDev, bool bias)
    c_Net& fullyConnectedLayer(int units, c_ActivationFunction act,
                               double stdDev, bool bias)
    c_Net& compressedLayer(int units, int params, c_ActivationFunction act,
                           string compression, double stdDev, bool bias)
    c_Net& extremeLayer(int units, c_ActivationFunction act, double stdDev,
                        bool bias)
    c_Net& convolutionalLayer(int featureMaps, int kernelRows, int kernelCols,
                              c_ActivationFunction act, double stdDev,
                              bool bias)
    c_Net& subsamplingLayer(int kernelRows, int kernelCols,
                            c_ActivationFunction act, double stdDev,
                            bool bias)
    c_Net& maxPoolingLayer(int kernelRows, int kernelCols, bool bias)
    c_Net& localReponseNormalizationLayer(double k, int n, double alpha,
                                          double beta, bool bias)
    c_Net& dropoutLayer(double dropoutProbability)
    c_Net& outputLayer(int units, c_ActivationFunction act, double stdDev)
    c_Net& compressedOutputLayer(int units, int params,
                                 c_ActivationFunction act,
                                 string& compression, double stdDev)
    c_Learner& trainingSet(c_MatrixXd& trainingInput, c_MatrixXd& trainingOutput)
    double error()
    c_VectorXd predict "operator()"(c_VectorXd x)
  c_Net *new_Net "new OpenANN::Net" ()
  void del_Net "delete" (c_Net *ptr)

cdef extern from "OpenANN/Convenience.h" namespace "OpenANN":
  void train(c_Net& net, string algorithm, c_ErrorFunction errorFunction,
             c_StoppingCriteria stop, bool reinitialize, bool dropout)

cdef class Net:
  cdef c_Net *thisptr
  cdef c_MatrixXd *inptr
  cdef c_MatrixXd *outptr
  cdef c_StoppingCriteria *stop
  cdef c_VectorXd *xptr
  cdef c_VectorXd *yptr

  def __cinit__(self):
    self.thisptr = new_Net()
    self.stop = new_StoppingCriteria()
    self.inptr = NULL
    self.outptr = NULL
    self.xptr = NULL
    self.yptr = NULL

  def __dealloc__(self):
    del_Net(self.thisptr)
    del_StoppingCriteria(self.stop)
    if self.inptr != NULL:
      del_MatrixXd(self.inptr)
    if self.outptr != NULL:
      del_MatrixXd(self.outptr)
    if self.xptr != NULL:
      del_VectorXd(self.xptr)
    if self.yptr != NULL:
      del_VectorXd(self.yptr)

  def __get_dims(self, shape, max_dim):
    shape_array = numpy.asarray(shape).flatten()
    assert len(shape_array) in range(1, 1+max_dim)
    dims = numpy.append(shape_array, numpy.ones(max_dim-len(shape_array)))
    return dims

  def __get_activation_function(self, act):
    return {"logistic" : LOGISTIC,
            "tanh" : TANH,
            "tanhscaled" : TANH_SCALED,
            "rectifier" : RECTIFIER,
            "linear" : LINEAR}[act]

  def __get_error_function(self, err):
    return {"sse" : SSE,
            "mse" : MSE,
            "ce" : CE}[err]

  def input_layer(self, shape, bias=True):
    dims = self.__get_dims(shape, 3)
    self.thisptr.inputLayer(dims[0], dims[1], dims[2], bias)
    return self

  def alpha_beta_filter_layer(self, delta_t, std_dev=0.05, bias=True):
    self.thisptr.alphaBetaFilterLayer(delta_t, std_dev, bias)
    return self

  def fully_connected_layer(self, units, act, std_dev=0.05, bias=True):
    self.thisptr.fullyConnectedLayer(units, self.__get_activation_function(act),
                                     std_dev, bias)
    return self

  def compressed_layer(self, units, params, act, compression, std_dev=0.05,
                       bias=True):
    cdef char* comp = compression
    self.thisptr.compressedLayer(units, params, self.__get_activation_function(act),
                                 string(comp), std_dev, bias)
    return self

  def extreme_layer(self, units, act, std_dev=5.0, bias=True):
    self.thisptr.extremeLayer(units, self.__get_activation_function(act),
                              std_dev, bias)
    return self

  def convolutional_layer(self, featureMaps, kernelRows, kernelCols, act,
                          std_dev=0.05, bias=True):
    self.thisptr.convolutionalLayer(featureMaps, kernelRows, kernelCols,
                                    self.__get_activation_function(act),
                                    std_dev, bias)
    return self

  def subsampling_layer(self, kernelRows, kernelCols, act, std_dev=0.05,
                        bias=True):
    self.thisptr.subsamplingLayer(kernelRows, kernelCols,
                                  self.__get_activation_function(act),
                                  std_dev, bias)
    return self

  def maxpooling_layer(self, kernelRows, kernelCols, bias=True):
    self.thisptr.maxPoolingLayer(kernelRows, kernelCols, bias)
    return self

  def local_response_normalization_layer(self, k, n, alpha, beta, bias=True):
    self.thisptr.localReponseNormalizationLayer(k, n, alpha, beta, bias)
    return self

  def dropout_layer(self, dropout_probability):
    self.thisptr.dropoutLayer(dropout_probability)
    return self

  def output_layer(self, units, act, std_dev=0.05):
    self.thisptr.outputLayer(units, self.__get_activation_function(act), std_dev)
    return self

  def compressed_output_layer(self, units, params, act, compression, std_dev=0.05):
    cdef char* comp = compression
    self.thisptr.compressedOutputLayer(units, params,
                                       self.__get_activation_function(act),
                                       string(comp), std_dev)
    return self

  def training_set(self, inputs, outputs):
    assert inputs.shape[1] == outputs.shape[1]
    if self.inptr != NULL:
      del_MatrixXd(self.inptr)
    self.inptr = new_MatrixXd(inputs.shape[0], inputs.shape[1])
    if self.outptr != NULL:
      del_MatrixXd(self.outptr)
    self.outptr = new_MatrixXd(outputs.shape[0], outputs.shape[1])
    self.__numpy_to_eigen_train(inputs, outputs)
    self.thisptr.trainingSet(deref(self.inptr), deref(self.outptr))
    return self

  def __numpy_to_eigen_train(self, num_in, num_out):
    rows = num_in.shape[0]
    cols = num_in.shape[1]
    idx = 0
    for c in range(cols):
      for r in range(rows):
        self.inptr.data()[idx] = num_in[r, c]
        idx += 1
    rows = num_out.shape[0]
    cols = num_out.shape[1]
    idx = 0
    for c in range(cols):
      for r in range(rows):
        self.outptr.data()[idx] = num_out[r, c]
        idx += 1

  def __stop_dict(self):
    return {"maximalFunctionEvaluations" : self.stop.maximalFunctionEvaluations,
            "maximalIterations" : self.stop.maximalIterations,
            "maximalRestarts" : self.stop.maximalRestarts,
            "minimalValue" : self.stop.minimalValue,
            "minimalValueDifferences" : self.stop.minimalValueDifferences,
            "minimalSearchSpaceStep" : self.stop.minimalSearchSpaceStep}

  def __stop_from_dict(self, d):
    self.stop.maximalFunctionEvaluations = d.get("maximalFunctionEvaluations",
        self.stop.maximalFunctionEvaluations)
    self.stop.maximalIterations = d.get("maximalIterations",
        self.stop.maximalIterations)
    self.stop.maximalRestarts = d.get("maximalRestarts",
        self.stop.maximalRestarts)
    self.stop.minimalValue = d.get("minimalValue", self.stop.minimalValue)
    self.stop.minimalValueDifferences = d.get("minimalValueDifferences",
        self.stop.minimalValueDifferences)
    self.stop.minimalSearchSpaceStep = d.get("minimalSearchSpaceStep",
        self.stop.minimalSearchSpaceStep)

  def train(self, algorithm, err, stop, reinitialize=True, dropout=False):
    self.__stop_from_dict(stop)
    cdef char* alg = algorithm
    train(deref(self.thisptr), string(alg), self.__get_error_function(err),
          deref(self.stop), reinitialize, dropout)

  def predict(self, X):
    if len(X.shape) == 2:
      Y = []
      for i in range(X.shape[1]):
        Y.append(self.__predict(X[:, i]))
      return numpy.asarray(Y).T
    else:
      return self.__predict(X)

  def __predict(self, x):
    self.__numpy_to_eigen_input(x)
    if self.yptr != NULL:
      del_VectorXd(self.yptr)
    self.yptr = new_VectorXd(self.thisptr.predict(deref(self.xptr)))
    return self.__eigen_to_numpy_output()

  def __numpy_to_eigen_input(self, x):
    if self.xptr != NULL:
      del_VectorXd(self.xptr)
    self.xptr = new_VectorXd(x.shape[0], 1)
    rows = x.shape[0]
    for r in range(rows):
      self.xptr.data()[r] = x[r]

  def __eigen_to_numpy_output(self):
    cdef int rows = self.yptr.rows()
    y = numpy.ndarray((rows,))
    for r in range(rows):
      y[r] = self.yptr.data()[r]
    return y

  def error(self):
    return self.thisptr.error()
