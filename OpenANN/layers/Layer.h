#pragma once

#include <Eigen/Dense>
#include <vector>

namespace OpenANN {

/**
 * Contains information about the output of a layer.
 */
class OutputInfo
{
public:
  //! Is there a bias in this layer, i. e. an input node that is always 1?
  bool bias;
  //! The dimensions of the output. There can be 1-3 dimensions.
  std::vector<int> dimensions;

  /**
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
  /**
   * Fill in the parameter pointers and parameter derivative pointers.
   * @param parameterPointers pointers to parameters
   * @param parameterDerivativePointers pointers to derivatives of parameters
   * @return information about the output of the layer
   */
  virtual OutputInfo initialize(std::vector<fpt*>& parameterPointers, std::vector<fpt*>& parameterDerivativePointers) = 0;
  /**
   * Initialize the parameters. This is usually called before each
   * optimization.
   */
  virtual void initializeParameters() = 0;
  /**
   * Forward propagation in this layer.
   * @param x pointer to input of the layer (with bias)
   * @param y returns a pointer to output of the layer
   */
  virtual void forwardPropagate(Vt* x, Vt*& y) = 0;
  /**
   * Backpropagation in this layer.
   * @param ein pointer to error signal of the higher layer
   * @param eout returns a pointer to error signal of the layer
   */
  virtual void backpropagate(Vt* ein, Vt*& eout) = 0;
};

}
