from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "Eigen/Dense" namespace "Eigen":
  cdef cppclass VectorXd:
    VectorXd()
    VectorXd(int rows)
    VectorXd(VectorXd&)
    double* data()
    int rows()
    double& get "operator()"(int rows)

  cdef cppclass MatrixXd:
    MatrixXd()
    MatrixXd(int rows, int cols)
    double& coeff(int row, int col)
    double* data()
    int rows()
    int cols()
    double& get "operator()"(int rows, int cols)

cdef extern from "OpenANN/io/Logger.h" namespace "OpenANN::Log":
  int& log_get_level "getLevel" ()

cdef extern from "OpenANN/Learner.h" namespace "OpenANN":
  cdef cppclass Learner

cdef extern from "OpenANN/layers/Layer.h" namespace "OpenANN":
  cdef cppclass Layer

cdef extern from "OpenANN/io/DataSet.h" namespace "OpenANN":
  cdef cppclass DataSet:
    int samples()
    int inputs()
    int outputs()
    VectorXd& getInstance(int i)
    VectorXd& getTarget(int i)
    void finishIteration(Learner& learner)

cdef extern from "OpenANN/io/DirectStorageDataSet.h" namespace "OpenANN":
  cdef cppclass DirectStorageDataSet(DataSet):
    DirectStorageDataSet(MatrixXd& input, MatrixXd& output)

cdef extern from "OpenANN/ActivationFunctions.h" namespace "OpenANN":
  cdef enum ActivationFunction:
    LOGISTIC
    TANH
    TANH_SCALED
    RECTIFIER
    LINEAR

cdef extern from "OpenANN/Net.h" namespace "OpenANN":
  cdef enum ErrorFunction:
    NO_E_DEFINED
    SSE
    MSE
    CE

cdef extern from "OpenANN/optimization/StoppingCriteria.h" namespace "OpenANN":
  cdef cppclass StoppingCriteria:
    StoppingCriteria()
    int maximalFunctionEvaluations
    int maximalIterations
    int maximalRestarts
    double minimalValue
    double minimalValueDifferences
    double minimalSearchSpaceStep


cdef extern from "OpenANN/optimization/Optimizable.h" namespace "OpenANN":
  cdef cppclass Optimizable:
    bool providesInitialization()
    void initialize()
    VectorXd currentParameters()
    double error()
    double error_from "error" (unsigned int i)
    bool providesGradient()
    VectorXd gradient_from "gradient" (unsigned int i)
    VectorXd gradient()


cdef extern from "OpenANN/optimization/Optimizer.h" namespace "OpenANN":
  cdef cppclass Optimizer:
    void setOptimizable(Optimizable& optimizable)
    void setStopCriteria(StoppingCriteria& sc)
    void optimize()
    VectorXd result()
    bool step()
    string name()


cdef extern from "OpenANN/optimization/MBSGD.h" namespace "OpenANN":
  cdef cppclass MBSGD(Optimizer):
    MBSGD(double learningRate, double momentum, int batchSize,
       double gamma, 
       double learningRateDecay, double minimalLearningRate, 
       double momentumGain, double maximalMomentum,
       double minGain, double maxGain)

cdef extern from "OpenANN/optimization/LMA.h" namespace "OpenANN":
  cdef cppclass LMA(Optimizer):
    LMA()

cdef extern from "OpenANN/Net.h" namespace "OpenANN":
  cdef cppclass Net(Optimizable):
    Net()
    Net& inputLayer(int dim1, int dim2, int dim3, bool bias)
    Net& alphaBetaFilterLayer(double deltaT, double stdDev, bool bias)
    Net& fullyConnectedLayer(int units, ActivationFunction act, double stdDev, bool bias)
    Net& compressedLayer(int units, int params, ActivationFunction act, string compression, double stdDev, bool bias)
    Net& extremeLayer(int units, ActivationFunction act, double stdDev, bool bias)
    Net& convolutionalLayer(int featureMaps, int kernelRows, int kernelCols, ActivationFunction act, double stdDev, bool bias)
    Net& subsamplingLayer(int kernelRows, int kernelCols, ActivationFunction act, double stdDev, bool bias)
    Net& maxPoolingLayer(int kernelRows, int kernelCols, bool bias)
    Net& localReponseNormalizationLayer(double k, int n, double alpha, double beta, bool bias)
    Net& outputLayer(int units, ActivationFunction act, double stdDev)
    Net& compressedOutputLayer(int units, int params, ActivationFunction act, string& compression, double stdDev)
    Net& dropoutLayer(double dropoutProbability) 
    Net& setErrorFunction(ErrorFunction errorFunction)
    Net& useDropout(bool activate)
    Net& addLayer(Layer *layer)
    unsigned int numberOflayer()
    VectorXd predict "operator()" (VectorXd& x)
    Learner& trainingSet(MatrixXd& inputs, MatrixXd& outputs)
    Learner& trainingSet(DataSet& dataset)

cdef extern from "OpenANN/io/LibSVM.h":
  int libsvm_load "OpenANN::LibSVM::load" (MatrixXd& input, MatrixXd& output, char *filename, int min_inputs)
  void save (MatrixXd& input, MatrixXd& output, char *filename)


