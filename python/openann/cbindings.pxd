from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

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

  cdef cppclass MatrixXi:
    MatrixXi()
    MatrixXi(int rows, int cols)
    int& coeff(int row, int col)
    int* data()
    int rows()
    int cols()
    int& get "operator()"(int rows, int cols)

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
    MSE
    CE

cdef extern from "<iostream>" namespace "std":
  cdef cppclass ostream
  ostream& write "operator<<" (ostream& os, char* str)

cdef extern from "OpenANN/io/Logger.h" namespace "OpenANN::Logger":
  cdef enum Target:
    NONE
    CONSOLE
    FILE
    APPEND_FILE

cdef extern from "OpenANN/io/Logger.h" namespace "OpenANN::Log":
  cdef enum LogLevel:
    DISABLED
    ERROR
    INFO
    DEBUG

cdef extern from "OpenANN/io/Logger.h" namespace "OpenANN":
  cdef cppclass Log:
    Log()
    ostream& get(LogLevel level, char* namespace = ?)

cdef extern from "OpenANN/io/Logger.h" namespace "OpenANN::Log":
  void setDisabled()
  void setError()
  void setInfo()
  void setDebug()


cdef extern from "OpenANN/layers/Layer.h" namespace "OpenANN":
  cdef cppclass OutputInfo:
    bool bias
    vector[int] dimensions
    int outputs()

  cdef cppclass Layer:
    OutputInfo initialize(vector[double*]& param, vector[double*] derivative)
    void initializeParameters()
    void updatedParameters()
    void forwardPropagate(VectorXd* x, VectorXd*& y, bool dropout)
    void backpropagate(VectorXd* ein, VectorXd*& eout)
    MatrixXd& getOutput()
    VectorXd getParameters()


cdef extern from "OpenANN/layers/SigmaPi.h" namespace "OpenANN::SigmaPi":
  cdef cppclass Constraint:
    double constrain "operator()" (int p1, int p2)
    double constrain "operator()" (int p1, int p2, int p3)
    double constrain "operator()" (int p1, int p2, int p3, int p4)

cdef extern from "OpenANN/layers/SigmaPi.h" namespace "OpenANN":
  cdef cppclass SigmaPi(Layer):
    SigmaPi(OutputInfo info, bool bias, ActivationFunction act, double stdDev)
    SigmaPi& secondOrderNodes(int numbers)
    SigmaPi& thirdOrderNodes(int numbers)
    SigmaPi& fourthOrderNodes(int numbers)
    SigmaPi& secondOrderNodes(int numbers, Constraint& constrain)
    SigmaPi& thirdOrderNodes(int numbers, Constraint& constrain)
    SigmaPi& fourthOrderNodes(int numbers, Constraint& constrain)

cdef extern from "OpenANN/layers/SigmaPiConstraints.h" namespace "OpenANN":
  cdef cppclass DistanceConstraint(Constraint):
    DistanceConstraint(long width, long height)
  cdef cppclass SlopeConstraint(Constraint):
    SlopeConstraint(long width, long height)
  cdef cppclass TriangleConstraint(Constraint):
    TriangleConstraint(long width, long height, double resolution)


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
    DirectStorageDataSet(MatrixXd* input, MatrixXd* output)

cdef extern from "OpenANN/io/LibSVM.h":
  int libsvm_load "OpenANN::LibSVM::load" (MatrixXd& input, MatrixXd& output,
                                           char *filename, int min_inputs)
  void save (MatrixXd& input, MatrixXd& output, char *filename)


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
    VectorXd& currentParameters()
    void setParameters(VectorXd& parameters)
    int dimension()
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
       double learningRateDecay, double minimalLearningRate, 
       double momentumGain, double maximalMomentum,
       double minGain, double maxGain)

cdef extern from "OpenANN/optimization/LMA.h" namespace "OpenANN":
  cdef cppclass LMA(Optimizer):
    LMA()

cdef extern from "OpenANN/optimization/CG.h" namespace "OpenANN":
  cdef cppclass CG(Optimizer):
    CG()

cdef extern from "OpenANN/optimization/LBFGS.h" namespace "OpenANN":
  cdef cppclass LBFGS(Optimizer):
    LBFGS(int m)

cdef extern from "OpenANN/Learner.h" namespace "OpenANN":
  cdef cppclass Learner(Optimizable):
    Learner& trainingSet(MatrixXd& input, MatrixXd& output)
    Learner& trainingSet(DataSet& dataset)
    MatrixXd predict "operator()" (MatrixXd& x)

cdef extern from "OpenANN/Net.h" namespace "OpenANN":
  cdef cppclass Net(Learner):
    Net()
    Net& inputLayer(int dim1, int dim2, int dim3)
    Net& alphaBetaFilterLayer(double deltaT, double stdDev)
    Net& fullyConnectedLayer(int units, ActivationFunction act, double stdDev,
                             bool bias)
    Net& restrictedBoltzmannMachineLayer(int H, int cdN, double stdDev,
                                         bool backprop)
    Net& compressedLayer(int units, int params, ActivationFunction act,
                         string compression, double stdDev, bool bias)
    Net& extremeLayer(int units, ActivationFunction act, double stdDev,
                      bool bias)
    Net& intrinsicPlasticityLayer(double targetMean, double stdDev)
    Net& convolutionalLayer(int featureMaps, int kernelRows, int kernelCols,
                            ActivationFunction act, double stdDev, bool bias)
    Net& subsamplingLayer(int kernelRows, int kernelCols,
                          ActivationFunction act, double stdDev, bool bias)
    Net& maxPoolingLayer(int kernelRows, int kernelCols)
    Net& localReponseNormalizationLayer(double k, int n, double alpha,
                                        double beta)
    Net& dropoutLayer(double dropoutProbability)
    Net& outputLayer(int units, ActivationFunction act, double stdDev)
    Net& compressedOutputLayer(int units, int params, ActivationFunction act,
                               string& compression, double stdDev)
    Net& addLayer(Layer *layer)
    Net& addOutputLayer(Layer *layer)

    Net& setRegularization(double l1Penalty, double l2Penalty,
                           double maxSquaredWeightNorm)
    Net& setErrorFunction(ErrorFunction errorFunction)
    Net& useDropout(bool activate)

    unsigned int numberOflayers()
    Layer& getLayer(unsigned int l)
    OutputInfo getOutputInfo(int l)
    DataSet* propagateDataSet(DataSet& dataSet, int l)

cdef extern from "OpenANN/SparseAutoEncoder.h" namespace "OpenANN":
  cdef cppclass SparseAutoEncoder(Learner):
    SparseAutoEncoder(int D, int H, double beta, double rho, double lmbda,
                      ActivationFunction act)
    MatrixXd getInputWeights()
    MatrixXd getOutputWeights()
    VectorXd reconstruct(VectorXd& x)

cdef extern from "OpenANN/Transformer.h" namespace "OpenANN":
  cdef cppclass Transformer:
    Transformer& fit(MatrixXd& X)
    MatrixXd transform(MatrixXd& X)

cdef extern from "OpenANN/Normalization.h" namespace "OpenANN":
  cdef cppclass Normalization(Transformer):
    Normalization()
    MatrixXd getMean()
    MatrixXd getStd()

cdef extern from "OpenANN/PCA.h" namespace "OpenANN":
  cdef cppclass PCA(Transformer):
    PCA(int components, bool whiten)
    VectorXd explainedVarianceRatio()

cdef extern from "OpenANN/Evaluation.h" namespace "OpenANN":
  double sse(Learner& learner, DataSet& dataSet)
  double mse(Learner& learner, DataSet& dataSet)
  double rmse(Learner& learner, DataSet& dataSet)
  double accuracy(Learner& learner, DataSet& dataSet)
  MatrixXi confusionMatrix(Learner& learner, DataSet& dataSet)
  int classificationHits(Learner& learner, DataSet& dataSet)
  void crossValidation(int folds, Learner& learner, DataSet& dataSet, Optimizer& opt)
