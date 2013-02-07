#pragma once

#include <Optimizable.h>
#include <Learner.h>
#include <layers/Layer.h>
#include <ActivationFunctions.h>
#include <StopCriteria.h>
#include <io/Logger.h>
#include <vector>

namespace OpenANN {

/**
 * @class DeepNetwork
 *
 * Deep neural network. You can specify many different types of layers and
 * choose the architecture almost arbitrary. But there are no shortcut
 * connections allowed!
 *
 * So far we implemented the following types of layers:
 *
 * - Input layer: adds a bias to the network's input. This layer must be
 *   present in the network.
 * - Output layer: this has to be the last layer and must be present in every
 *   network. It is fully connected.
 * - Compressed output layer: this is an alternative to the output layer. It
 *   is a fully connected output layer.
 * - FullyConnected layer: each neuron is connected to each neuron of the
 *   previous layer.
 * - Compressed layer: fully connected layer. The I incoming weights of a
 *   neuron are represented by M (usually M < I) parameters.
 * - Convolutional layer: consists of a number of 2-dimensional feature maps.
 *   Each feature map is connected to each feature map of the previous layer.
 *   The activations are computed by applying a parametrizable convolution,
 *   i. e. this kind of layer uses weight sharing and sparse connections to
 *   reduce the number of weights in comparison to fully connected layers.
 * - Subsampling layer: these will be used to quickly reduce the number of
 *   nodes after a convolution and obtain little translation invarianc. A
 *   non-overlapping group of nodes is summed up, multiplied with a weight and
 *   added to a learnable bias to obtain the activation of a neuron. This is
 *   sometimes called average pooling.
 * - MaxPooling layer: this is an alternative to subsampling layers and works
 *   usually better. Instead of the sum it computes the maximum of a group and
 *   has no learnable weights or biases.
 * - AlphaBetaFilter layer: this is a recurrent layer that estimates the
 *   position and velocity of the inputs from the noisy observation of the
 *   positions. Usually we need this layer for partially observable markov
 *   decision processes in reinforcement learning.
 */
class DeepNetwork : public Optimizable, public Learner
{
public:
  enum ErrorFunction
  {
    NO_E_DEFINED,
    SSE, //!< Sum of squared errors and identity (regression)
    MSE, //!< Mean of squared errors and identity (regression)
    CE   //!< Cross entropy and softmax (classification)
  };

  enum Training
  {
    NOT_INITIALIZED,
    BATCH_CMAES,  //!< Covariance Matrix Adaption Evolution Strategies
    BATCH_LMA,    //!< Levenberg-Marquardt Algorithm
    BATCH_SGD,    //!< Stochastic Gradient Descent
    MINIBATCH_SGD //!< Mini-Batch Stochastic Gradient Descent
  };

private:
  Logger debugLogger;
  std::vector<OutputInfo> infos;
  std::vector<Layer*> layers;
  std::vector<fpt*> parameters;
  std::vector<fpt*> derivatives;
  DataSet* dataSet;
  DataSet* testDataSet;
  bool deleteDataSet, deleteTestSet;
  ErrorFunction errorFunction;
  bool dropout;

  bool initialized;
  int P, N, L;
  Vt parameterVector;
  Vt tempInput, tempOutput, tempError, tempGradient;

  void initializeNetwork();

public:
  DeepNetwork(ErrorFunction errorFunction);
  virtual ~DeepNetwork();

  /**
   * Add an input layer.
   * @param dim1 first dimension of the input, e. g. number of color channels
   *             of an image
   * @param dim2 second dimension, e. g. number of rows of an image
   * @param dim3 third dimension, e. g. number of columns of an image
   * @param bias add bias term
   * @param dropoutProbability probability of dropout during training, a
   *        reasonable value is usually between 0 and 0.5
   * @return this for chaining
   */
  DeepNetwork& inputLayer(int dim1, int dim2 = 1, int dim3 = 1,
                          bool bias = true, fpt dropoutProbability = 0.0);
  /**
   * Add a alpha-beta filter layer.
   * @param deltaT temporal difference between two steps
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @param bias add bias term
   * @return this for chaining
   */
  DeepNetwork& alphaBetaFilterLayer(fpt deltaT, fpt stdDev = (fpt) 0.05, bool bias = true);
  /**
   * Add a fully connected hidden layer.
   * @param units number of nodes (neurons)
   * @param act activation function
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @param bias add bias term
   * @param dropoutProbability probability of dropout during training, a
   *        reasonable value is usually between 0 and 0.5
   * @return this for chaining
   */
  DeepNetwork& fullyConnectedLayer(int units, ActivationFunction act,
                                   fpt stdDev = (fpt) 0.05, bool bias = true,
                                   fpt dropoutProbability = 0.0);
  /**
   * Add a compressed fully connected hidden layer.
   * @param units number of nodes (neurons)
   * @param params number of parameters to represent the incoming weights of
   *               a neuron in this layer
   * @param act activation function
   * @param compression type of compression matrix, possible values are
   *        dct, gaussian, sparse, average, edge
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @param bias add bias term
   * @param dropoutProbability probability of dropout during training, a
   *        reasonable value is usually between 0 and 0.5
   * @return this for chaining
   */
  DeepNetwork& compressedLayer(int units, int params, ActivationFunction act,
                               const std::string& compression,
                               fpt stdDev = (fpt) 0.05, bool bias = true,
                               fpt dropoutProbability = 0.0);
  /**
   * Add a convolutional layer.
   * @param featureMaps number of feature maps
   * @param kernelRows number of kernel rows
   * @param kernelCols number of kernel columns
   * @param act activation function
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @param bias add bias term
   * @return this for chaining
   */
  DeepNetwork& convolutionalLayer(int featureMaps, int kernelRows,
                                  int kernelCols, ActivationFunction act,
                                  fpt stdDev = (fpt) 0.05, bool bias = true);
  /**
   * Add a subsampling layer.
   * @param kernelRows number of kernel rows
   * @param kernelCols number of kernel columns
   * @param act activation function
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @param bias add bias term
   * @return this for chaining
   */
  DeepNetwork& subsamplingLayer(int kernelRows, int kernelCols,
                                ActivationFunction act,
                                fpt stdDev = (fpt) 0.05, bool bias = true);
  /**
   * Add a max-pooling layer.
   * @param kernelRows number of kernel rows
   * @param kernelCols number of kernel columns
   * @param bias add bias term
   * @return this for chaining
   */
  DeepNetwork& maxPoolingLayer(int kernelRows, int kernelCols, bool bias = true);
  /**
   * Add a fully connected output layer. This will initialize the network.
   * @param units number of nodes (neurons)
   * @param act activation function
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @return this for chaining
   */
  DeepNetwork& outputLayer(int units, ActivationFunction act,
                           fpt stdDev = (fpt) 0.05);
  /**
   * Add a compressed output layer. This will initialize the network.
   * @param units number of nodes (neurons)
   * @param params number of parameters to represent the incoming weights of
   *               a neuron in this layer
   * @param act activation function
   * @param compression type of compression matrix, possible values are
   *        dct, gaussian, sparse, average, edge
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @return this for chaining
   */
  DeepNetwork& compressedOutputLayer(int units, int params,
                                     ActivationFunction act,
                                     const std::string& compression,
                                     fpt stdDev = (fpt) 0.05);

  virtual Learner& trainingSet(Mt& trainingInput, Mt& trainingOutput);
  virtual Learner& trainingSet(DataSet& trainingSet);
  virtual DeepNetwork& testSet(Mt& testInput, Mt& testOutput);
  virtual DeepNetwork& testSet(DataSet& testDataSet);
  Vt train(Training algorithm, StopCriteria stop, bool reinitialize = true,
           bool dropout = false);
  virtual void finishedIteration();

  virtual Vt operator()(const Vt& x);
  virtual unsigned int dimension();
  virtual unsigned int examples();
  virtual Vt currentParameters();
  virtual void setParameters(const Vt& parameters);
  virtual bool providesInitialization();
  virtual void initialize();
  virtual fpt error(unsigned int i);
  virtual fpt error();
  virtual bool providesGradient();
  virtual Vt gradient(unsigned int i);
  virtual Vt gradient();
  virtual void VJ(Vt& values, Mt& jacobian);
  virtual bool providesHessian();
  virtual Mt hessian();
};

}
