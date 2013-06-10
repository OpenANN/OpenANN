#ifndef OPENANN_NET_H_
#define OPENANN_NET_H_

#include <OpenANN/Learner.h>
#include <OpenANN/ActivationFunctions.h>
#include <OpenANN/Regularization.h>
#include <OpenANN/layers/Layer.h>
#include <vector>

namespace OpenANN
{

/**
 * @enum ErrorFunction
 *
 * Error function that will be minimized.
 */
enum ErrorFunction
{
  NO_E_DEFINED,
  SSE, //!< Sum of squared errors (regression, two classes)
  MSE, //!< Mean of squared errors (regression, two classes)
  CE   //!< Cross entropy and softmax (multiple classes)
};

/**
 * @class Net
 *
 * Feedforward multilayer neural network.
 *
 * You can specify many different types of layers and choose the architecture
 * almost arbitrary.
 */
class Net : public Learner
{
  std::vector<OutputInfo> infos;
  std::vector<Layer*> layers;
  std::vector<double*> parameters;
  std::vector<double*> derivatives;
  Regularization regularization;
  ErrorFunction errorFunction;
  bool dropout;

  bool initialized;
  int P, L;
  Eigen::VectorXd parameterVector, tempGradient;
  Eigen::MatrixXd tempInput, tempOutput, tempError;

  void initializeNetwork();

public:
  Net();
  virtual ~Net();

  /**
   * @name Architecture Definition
   * These functions must be called to define the architecture of the network.
   */
  ///@{
  /**
   * Add an input layer.
   * @param dim1 first dimension of the input, e. g. number of color channels
   *             of an image
   * @param dim2 second dimension, e. g. number of rows of an image
   * @param dim3 third dimension, e. g. number of columns of an image
   * @return this for chaining
   */
  Net& inputLayer(int dim1, int dim2 = 1, int dim3 = 1);
  /**
   * Add a alpha-beta filter layer.
   * @param deltaT temporal difference between two steps
   * @param stdDev standard deviation of the Gaussian distributed initial
   *               weights
   * @return this for chaining
   */
  Net& alphaBetaFilterLayer(double deltaT, double stdDev = 0.05);
  /**
   * Add a fully connected hidden layer.
   * @param units number of nodes (neurons)
   * @param act activation function
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @param bias add bias term
   * @return this for chaining
   */
  Net& fullyConnectedLayer(int units, ActivationFunction act,
                           double stdDev = 0.05, bool bias = true);
  /**
   * Add a layer that contains an RBM.
   * @param H number of nodes (neurons)
   * @param cdN number of gibbs sampling steps for pretraining
   * @param stdDev standard deviation of the Gaussian distributed initial
   *               weights
   * @param backprop finetune weights with backpropagation
   * @return this for chaining
   */
  Net& restrictedBoltzmannMachineLayer(int H, int cdN = 1,
                                       double stdDev = 0.01,
                                       bool backprop = true);
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
   * @return this for chaining
   */
  Net& compressedLayer(int units, int params, ActivationFunction act,
                       const std::string& compression, double stdDev = 0.05,
                       bool bias = true);
  /**
   * Add a fully connected hidden layer with fixed weights.
   * @param units number of nodes (neurons)
   * @param act activation function
   * @param stdDev standard deviation of the Gaussian distributed initial weights
   * @param bias add bias term
   * @return this for chaining
   */
  Net& extremeLayer(int units, ActivationFunction act, double stdDev = 5.0,
                    bool bias = true);
  /**
   * Add an intrinsic plasticity layer that is able to learn the parameters of
   * a logistic activation function.
   * @param targetMean desired mean of the output distribution, must be within
   *                   [0, 1], e.g. 0.2
   * @param stdDev standard deviation of the Gaussian distributed initial
   *               biases
   * @return this for chaining
   */
  Net& intrinsicPlasticityLayer(double targetMean, double stdDev = 1.0);
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
  Net& convolutionalLayer(int featureMaps, int kernelRows, int kernelCols,
                          ActivationFunction act, double stdDev = 0.05,
                          bool bias = true);
  /**
   * Add a subsampling layer.
   * @param kernelRows number of kernel rows
   * @param kernelCols number of kernel columns
   * @param act activation function
   * @param stdDev standard deviation of the Gaussian distributed initial
   *               weights
   * @param bias add bias term
   * @return this for chaining
   */
  Net& subsamplingLayer(int kernelRows, int kernelCols,
                        ActivationFunction act, double stdDev = 0.05,
                        bool bias = true);
  /**
   * Add a max-pooling layer.
   * @param kernelRows number of kernel rows
   * @param kernelCols number of kernel columns
   * @return this for chaining
   */
  Net& maxPoolingLayer(int kernelRows, int kernelCols);
  /**
   * Add a local response normalization layer.
   * \f$ y^i_{rc} = x^i_{rc} / \left( k +
   *     \alpha \sum_{j=max(0, i-n/2)}^{min(N-1, i+n/2)}
   *     x^j_{rc} \right)^{\beta} \f$
   * @param k hyperparameter, k >= 1, e.g. 1 or 2
   * @param n number of adjacent feature maps
   * @param alpha controls strength of inhibition, alpha > 0, e.g. 1e-4
   * @param beta controls strength of inhibition, beta > 0, e.g. 0.75
   * @return this for chaining
   */
  Net& localReponseNormalizationLayer(double k, int n, double alpha,
                                      double beta);
  /**
   * Add a dropout layer.
   * @param dropoutProbability probability of suppression during training
   */
  Net& dropoutLayer(double dropoutProbability);
  /**
   * Add a fully connected output layer. This will initialize the network.
   * @param units number of nodes (neurons)
   * @param act activation function
   * @param stdDev standard deviation of the Gaussian distributed initial
   *               weights
   * @param bias add bias term
   * @return this for chaining
   */
  Net& outputLayer(int units, ActivationFunction act, double stdDev = 0.05,
                   bool bias = true);
  /**
   * Add a compressed output layer. This will initialize the network.
   * @param units number of nodes (neurons)
   * @param params number of parameters to represent the incoming weights of
   *               a neuron in this layer
   * @param act activation function
   * @param compression type of compression matrix, possible values are
   *        dct, gaussian, sparse, average, edge
   * @param stdDev standard deviation of the Gaussian distributed initial
   *               weights
   * @param bias add bias term
   * @return this for chaining
   */
  Net& compressedOutputLayer(int units, int params, ActivationFunction act,
                             const std::string& compression,
                             double stdDev = 0.05, bool bias = true);
  /**
   * Add a new layer to this deep neural network.
   * Never free/delete the added layer outside of this class.
   * Its cleaned up by Net's destructor automatically.
   * @param layer pointer to an instance that implements the Layer interface
   * @return this for chaining
   */
  Net& addLayer(Layer* layer);
  /**
   * Add a new output layer to this deep neural network.
   * Never free/delete the added layer outside of this class.
   * Its cleaned up by Net's destructor automatically.
   * @param layer pointer to an instance that implements the Layer interface
   * @return this for chaining
   */
  Net& addOutputLayer(Layer* layer);
  ///@}

  /**
   * @name Access Internal Structure
   */
  ///@{
  /**
   * Request number of layers.
   * @return number of layers in this neural network
   */
  unsigned int numberOflayers();
  /**
   * Access layer.
   * @param l index of layer
   * @return l-th layer
   */
  Layer& getLayer(unsigned int l);
  /**
   * Request information about output of a given layer.
   * @param l index of the layer
   * @return output information
   */
  OutputInfo getOutputInfo(unsigned int l);
  /**
   * Propagate data set through the first l layers.
   * @param dataSet original dataset
   * @param l index of the layer
   * @return transformed data set (has to be deleted manually)
   */
  DataSet* propagateDataSet(DataSet& dataSet, int l);
  ///@}

  /**
   * @name Optimization Contol
   */
  ///@{
  Net& setRegularization(double l1Penalty = 0.0, double l2Penalty = 0.0,
                         double maxSquaredWeightNorm = 0.0);
  /**
   * Set the error function.
   * @param errorFunction error function
   * @return this for chaining
   */
  Net& setErrorFunction(ErrorFunction errorFunction);
  /**
   * Toggle dropout.
   * @param activate turn dropout on or off
   * @return this for chaining
   */
  Net& useDropout(bool activate = true);
  ///@}

  /**
   * @name Inherited Functions
   */
  ///@{
  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& x);
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd& X);
  virtual unsigned int dimension();
  virtual const Eigen::VectorXd& currentParameters();
  virtual void setParameters(const Eigen::VectorXd& parameters);
  virtual bool providesInitialization();
  virtual void initialize();
  virtual unsigned int examples();
  virtual double error(unsigned int i);
  virtual double error();
  virtual bool providesGradient();
  virtual Eigen::VectorXd gradient(unsigned int n);
  virtual Eigen::VectorXd gradient();
  virtual void errorGradient(int n, double& value, Eigen::VectorXd& grad);
  virtual void errorGradient(double& value, Eigen::VectorXd& grad);
  virtual void errorGradient(std::vector<int>::const_iterator startN,
                             std::vector<int>::const_iterator endN,
                             double& value, Eigen::VectorXd& grad);
  virtual void finishedIteration();
  ///@}
private:
  void forwardPropagate();
  void backpropagate();
  double generalErrorGradient(bool computeError, Eigen::VectorXd& g, int n = -1);
};

} // namespace OpenANN

#endif // OPENANN_NET_H

