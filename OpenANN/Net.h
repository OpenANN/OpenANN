#pragma once

#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/Learner.h>
#include <OpenANN/layers/Layer.h>
#include <OpenANN/ActivationFunctions.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <vector>

namespace OpenANN {

enum ErrorFunction
{
  NO_E_DEFINED,
  SSE, //!< Sum of squared errors (regression, two classes)
  MSE, //!< Mean of squared errors (regression, two classes)
  CE   //!< Cross entropy and softmax (multiple classes)
};

enum Training
{
  NOT_INITIALIZED,
  BATCH_CMAES,  //!< Covariance Matrix Adaption Evolution Strategies (IPOPCMAES)
  BATCH_LMA,    //!< Levenberg-Marquardt Algorithm (LMA)
  MINIBATCH_SGD //!< Mini-Batch Stochastic Gradient Descent (MBSGD)
};

/**
 * @class Net
 *
 * Feedforward multilayer neural network. You can specify many different types
 * of layers and choose the architecture almost arbitrary. But there are no
 * shortcut connections allowed!
 *
 * So far the following types of layers are implemented:
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
 * - Extreme layer: fully connected layer with fixed random weights.
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
 * - LocalResponseNormalization layer: lateral inhibition of neurons at the
 *   same positions in adjacent feature maps.
 * - AlphaBetaFilter layer: this is a recurrent layer that estimates the
 *   position and velocity of the inputs from the noisy observation of the
 *   positions. Usually we need this layer for partially observable markov
 *   decision processes in reinforcement learning.
 */
class Net : public Optimizable, public Learner
{
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
  Net();
  virtual ~Net();

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
  Net& inputLayer(int dim1, int dim2 = 1, int dim3 = 1, bool bias = true,
                  fpt dropoutProbability = 0.0);
  /**
   * Add a alpha-beta filter layer.
   * @param deltaT temporal difference between two steps
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @param bias add bias term
   * @return this for chaining
   */
  Net& alphaBetaFilterLayer(fpt deltaT, fpt stdDev = (fpt) 0.05,
                            bool bias = true);
  /**
   * Add a fully connected hidden layer.
   * @param units number of nodes (neurons)
   * @param act activation function
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @param bias add bias term
   * @param dropoutProbability probability of dropout during training, a
   *        reasonable value is usually between 0 and 0.5
   * @param maxSquaredWeightNorm when training with dropout it is helpful to
   *        explore with a high learning rate and constrain the incoming
   *        weights of a neuron to have a maximum norm
   * @return this for chaining
   */
  Net& fullyConnectedLayer(int units, ActivationFunction act,
                                   fpt stdDev = (fpt) 0.05, bool bias = true,
                                   fpt dropoutProbability = 0.0,
                                   fpt maxSquaredWeightNorm = 0.0);
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
  Net& compressedLayer(int units, int params, ActivationFunction act,
                               const std::string& compression,
                               fpt stdDev = (fpt) 0.05, bool bias = true,
                               fpt dropoutProbability = 0.0);
  /**
   * Add a fully connected hidden layer with fixed weights.
   * @param units number of nodes (neurons)
   * @param act activation function
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @param bias add bias term
   * @return this for chaining
   */
  Net& extremeLayer(int units, ActivationFunction act,
                            fpt stdDev = (fpt) 5.0, bool bias = true);
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
  Net& convolutionalLayer(int featureMaps, int kernelRows,
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
  Net& subsamplingLayer(int kernelRows, int kernelCols,
                                ActivationFunction act,
                                fpt stdDev = (fpt) 0.05, bool bias = true);
  /**
   * Add a max-pooling layer.
   * @param kernelRows number of kernel rows
   * @param kernelCols number of kernel columns
   * @param bias add bias term
   * @return this for chaining
   */
  Net& maxPoolingLayer(int kernelRows, int kernelCols, bool bias = true);
  /**
   * Add a local response normalization layer.
   * \f$ y^i_{rc} = x^i_{rc} / \left( k +
   *     \alpha \sum_{j=max(0, i-n/2)}^{min(N-1, i+n/2)}
   *     x^j_{rc} \right)^{\beta} \f$
   * @param k hyperparameter, k >= 1, e.g. 1 or 2
   * @param n number of adjacent feature maps
   * @param alpha controls strength of inhibition, alpha > 0, e.g. 1e-4
   * @param beta controls strength of inhibition, beta > 0, e.g. 0.75
   * @param bias add bias term
   * @return this for chaining
   */
  Net& localReponseNormalizationLayer(fpt k, int n, fpt alpha, fpt beta,
                                      bool bias = true);
  /**
   * Add a fully connected output layer. This will initialize the network.
   * @param units number of nodes (neurons)
   * @param act activation function
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @return this for chaining
   */
  Net& outputLayer(int units, ActivationFunction act, fpt stdDev = (fpt) 0.05);
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
  Net& compressedOutputLayer(int units, int params, ActivationFunction act,
                             const std::string& compression,
                             fpt stdDev = (fpt) 0.05);
  /** 
   * Add a new layer to this deep neural network. 
   * Never free/delete the added layer outside of this class. 
   * Its cleaned up by Net's destructor automatically.
   * @param layer pointer to an instance that implements the Layer interface
   * @return this for chaining
   */
  Net& addLayer(Layer* layer);

  unsigned int numberOflayers();
  Layer& getLayer(unsigned int l);
  OutputInfo getOutputInfo(unsigned int l);
  Net& setErrorFunction(ErrorFunction errorFunction);
  virtual Learner& trainingSet(Mt& trainingInput, Mt& trainingOutput);
  virtual Learner& trainingSet(DataSet& trainingSet);
  virtual Net& testSet(Mt& testInput, Mt& testOutput);
  virtual Net& testSet(DataSet& testDataSet);
  Vt train(Training algorithm, ErrorFunction errorFunction, StoppingCriteria stop,
           bool reinitialize = true, bool dropout = false);

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
  virtual fpt errorFromDataSet(DataSet& dataset);
  virtual bool providesGradient();
  virtual Vt gradient(unsigned int i);
  virtual Vt gradient();
  virtual void VJ(Vt& values, Mt& jacobian);
  virtual bool providesHessian();
  virtual Mt hessian();
};

}
