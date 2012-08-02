#pragma once

#include <MLPImplementation.h>
#include <Learner.h>
#include <Optimizable.h>
#include <Optimizer.h>
#include <StopCriteria.h>
#include <io/DataSet.h>
#include <io/Logger.h>
#include <Test/Stopwatch.h>

namespace OpenANN {

class MLP : public Learner, public Optimizable
{
public:
  enum ErrorFunction
  {
    NO_E_DEFINED,
    SSE, //!< Sum of squared errors and identity (regression)
    MSE, //!< Mean of squared errors and identity (regression)
    CE   //!< Cross entropy and softmax (classification)
  };

  enum ActivationFunction
  {
    NO_ACT_FUN = -1,
    SIGMOID = 0,  //!< Logistic sigmoid, range [0,1]
    TANH = 1,     //!< Tangens Hyperbolicus, range [-1,1]
    ID = 2,       //!< Identity function (output activation function for regression)
    STANH = 3,    //!< Scaled Tangens Hyperbolicus
    SM = 4        //!< Softmax (output activation function for multi-class classification)
  };

  enum Training
  {
    NOT_INITIALIZED,
    BATCH_CMAES,  //!< Covariance Matrix Adaption Evolution Strategies
    BATCH_LMA,    //!< Levenberg-Marquardt Algorithm
    BATCH_SGD     //!< Stochastic Gradient Descent
  };

  /** Layer initialization state. */
  enum LayerState
  {
    NO_LAYER,
    INPUT_LAYER,
    HIDDEN_LAYER_FINISHED,
    OUTPUT_LAYER_FINISHED
  } layerState;

  /** Overall initialization state. */
  enum InitializationState
  {
    UNINITIALIZED,
    WEIGHTS_INIT,
    INITIALIZED,
    NUMBER_OF_INITIALIZATION_STATES
  } initializationState;

  /**
   * Chosen error function and output activation function combination.
   */
  ErrorFunction errorFunction;

  /** The actual MLP implementation. */
  MLPImplementation mlp;
  DataSet* trainingData;
  DataSet* testData;
  bool deleteDataSetOnDestruction;
  /** Optimizer. */
  Optimizer* optimizer;
  bool reinitialize;
  int iteration;
  Stopwatch sw;
  /** Training set size. */
  int N;
  Logger errorLogger;
  Logger parameterLogger;
  std::string configuration;

private: // temporary objects, avoiding memory allocations
  Vt temporaryGradient;
  Vt temporaryOutput;

public:
  /**
   * It is not sufficient to call this constructor. You have to specify at
   * least an output layer.
   * @param errorTarget target for error logger (after each iteration)
   * @param parameterTarget target for parameter logger (after each iteration)
   */
  MLP(Logger::Target errorTarget = Logger::CONSOLE, Logger::Target parameterTarget = Logger::NONE);
  virtual ~MLP();
  /**
   * Deactivates bias.
   */
  MLP& noBias();
  /**
   * Defines the dimension of a one-dimensional input.
   * @param units number of input components
   */
  MLP& input(int units);
  /**
   * Defines the dimensions of a two-dimensional input.
   * @param rows number of input rows
   * @param cols number of input columns
   */
  MLP& input(int rows, int cols);
  /**
   * Defines the dimensions of a three-dimensional input.
   * @param arrays two-dimensional input arrays, e. g. color channels
   * @param rows input rows
   * @param cols input columns
   */
  MLP& input(int arrays, int rows, int cols);
  /**
   * Add a convolutional layer.
   * @param featureMaps number of feature maps
   * @param kernelRows filter kernel rows
   * @param kernelCols filter kernel columns
   * @param a activation function
   */
  MLP& convolutionalLayer(int featureMaps, int kernelRows, int kernelCols,
      ActivationFunction a = NO_ACT_FUN);
  /**
   * Defines a fully connected hidden layer.
   * @param units number of nodes
   * @param a activation function
   * @param numberOfParameters number of compression parameters per node (set to
   *                           -1 for now compression)
   * @param compression Type of compression matrix. ("dct", "random")
   */
  MLP& fullyConnectedHiddenLayer(int units, ActivationFunction a = NO_ACT_FUN,
      int numberOfParameters = -1, std::string compression = "dct");
  /**
   * Defines a fully connected hidden layer with two-dimensional compression of the incoming weights.
   * @param units number of nodes
   * @param a activation function
   * @param parameterRows number of compression parameters per row
   * @param parameterCols number of compression parameters per column
   */
  MLP& fullyConnectedHiddenLayer(int units, ActivationFunction a,
      int parameterRows, int parameterCols);
  /**
   * Defines a fully connected hidden layer with three-dimensional compression
   * of the incoming weights.
   * @param units number of nodes
   * @param a activation function
   * @param parametersX number of compression parameters in the first dimension
   * @param parametersY number of compression parameters in the second dimension
   * @param parametersZ number of compression parameters in the third dimension
   * @param t mapping from indices of first and second input components to third
   *          input argument of third dimension compression function.
   */
  MLP& fullyConnectedHiddenLayer(int units, ActivationFunction a, int parametersX,
      int parametersY, int parametersZ, const std::vector<std::vector<fpt> >& t);
  /**
   * Defines a fully connected output vector.
   * @param units number of nodes
   * @param e error function (loss function)
   * @param a activation function
   * @param numberOfParameters number of compression parameters per node (set to
   *                           -1 for now compression)
   * @param compression Type of compression matrix. ("dct", "random")
   */
  MLP& output(int units, ErrorFunction e = SSE, ActivationFunction a = NO_ACT_FUN,
      int numberOfParameters = -1, std::string compression = "dct");
  /**
   * Defines a fully connected output vector with two-dimensional compression of
   * the incoming weights.
   * @param units number of nodes
   * @param e error function (loss function)
   * @param a activation function
   * @param parameterRows number of compression parameters per row
   * @param parameterCols number of compression parameters per column
   */
  MLP& output(int units, ErrorFunction e, ActivationFunction a, int parameterRows,
      int parameterCols);
  /**
   * Defines a fully connected output vector with three-dimensional compression
   * of the incoming weights.
   * @param units number of nodes
   * @param e error function (loss function)
   * @param a activation function
   * @param parametersX number of compression parameters in the first dimension
   * @param parametersY number of compression parameters in the second dimension
   * @param parametersZ number of compression parameters in the third dimension
   * @param t mapping from indices of first and second input components to third
   *          input argument of third dimension compression function.
   */
  MLP& output(int units, ErrorFunction e, ActivationFunction a, int parametersX,
      int parametersY, int parametersZ, const std::vector<std::vector<fpt> >& t);
private:
  MLP& makeOutputLayer(int units, ErrorFunction e, ActivationFunction a);
  MLP& compressWith(int parameterRows, int parameterCols);
  MLP& compressWith(int parametersX, int parametersY, int parametersZ, const std::vector<std::vector<fpt> >& t);
  MLP& finishInitialization();
public:
  virtual Learner& trainingSet(Mt& trainingInput, Mt& trainingOutput);
  virtual Learner& trainingSet(DataSet& trainingSet);
  /**
   * Set the current test set.
   * @param testInput input vectors, each instance should be in a new column
   * @param testOutput output vectors, each instance should be in a new column
   */
  MLP& testSet(Mt& testInput, Mt& testOutput);
  /**
   * Set the current test set.
   * @param testSet custom test set
   */
  MLP& testSet(DataSet& testSet);
  /**
   * Set the optimization algorithm.
   * @param training optimization algorithm
   * @param reinit should the optimizer be able to reinitialize the weights?
   */
  MLP& training(Training training, bool reinit = true);
  /**
   * Train the network.
   * @param stop stopping criteria for batch training
   */
  Vt fit(StopCriteria stop = StopCriteria::defaultValue);
  /**
   * @return dimension of the parameter vector
   */
  virtual unsigned int dimension();
  /**
   * @return error on the training set
   */
  virtual fpt error();
  virtual bool providesGradient();
  virtual Vt gradient();
  virtual bool providesHessian();
  virtual Mt hessian();
  virtual bool providesInitialization();
  virtual void initialize();
  virtual void setParameters(const Vt& parameters);
  virtual Vt currentParameters();
  virtual unsigned int examples();
  virtual fpt error(unsigned int i);
  virtual Vt gradient(unsigned int i);
  virtual Vt operator()(const Vt& x);
  virtual int operator()(const Vt& x, Vt& fvec);
  virtual int df(const Vt& x, Vt& fjac);
  virtual void VJ(Vt& values, Mt& jacobian);
  virtual void finishedIteration();
};

}
