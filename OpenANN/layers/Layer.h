#ifndef OPENANN_LAYERS_LAYER_H_
#define OPENANN_LAYERS_LAYER_H_

#include <Eigen/Dense>
#include <vector>

namespace OpenANN
{

/**
 * @class OutputInfo
 *
 * Provides information about the output of a layer.
 */
class OutputInfo
{
public:
  //! The dimensions of the output. There can be 1-3 dimensions.
  std::vector<int> dimensions;

  /**
   * Get number of outputs.
   * @return number of output nodes
   */
  int outputs();
};

/**
 * @class Layer
 *
 * Interface that has to be implemented by all layers of a neural network
 * that can be trained with backpropagation.
 */
class Layer
{
public:
  virtual ~Layer() {}
  /**
   * Fill in the parameter pointers and parameter derivative pointers.
   * @param parameterPointers pointers to parameters
   * @param parameterDerivativePointers pointers to derivatives of parameters
   * @return information about the output of the layer
   */
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers) = 0;
  /**
   * Initialize the parameters. This is usually called before each
   * optimization.
   */
  virtual void initializeParameters() = 0;
  /**
   * Generate internal parameters from externally visible parameters. This is
   * usually called after each parameter update.
   */
  virtual void updatedParameters() = 0;
  /**
   * Forward propagation in this layer.
   * @param x pointer to input of the layer (with bias)
   * @param y returns a pointer to output of the layer
   * @param dropout enable dropout for regularization
   */
  virtual void forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                                bool dropout) = 0;
  /**
   * Backpropagation in this layer.
   * @param ein pointer to error signal of the higher layer
   * @param eout returns a pointer to error signal of the layer (derivative of
   *             the error with respect to the input)
   * @param backpropToPrevious backpropagate errors to previous layers
   */
  virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,
                             bool backpropToPrevious) = 0;
  /**
   * Output after last forward propagation.
   * @return output
   */
  virtual Eigen::MatrixXd& getOutput() = 0;
  /**
   * Get the current values of parameters (weights, biases, ...).
   * @return parameters
   */
  virtual Eigen::VectorXd getParameters() = 0;
};

} // namespace OpenANN

#endif // OPENANN_LAYERS_LAYER_H_
