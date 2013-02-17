#pragma once

#include <layers/Layer.h>
#include <ActivationFunctions.h>

namespace OpenANN {

/**
 * @class FullyConnected
 *
 * Fully connected layer.
 *
 * Each neuron in the previous layer is taken as input for each neuron of this
 * layer. Forward propagation is usually done by \f$ a = W \cdot x, y_j =
 * g(a_j) \f$, where \f$ a_j \f$ is the activation of the jth neuron, \f$ y_j
 * \f$ its output, \f$ g \f$ a typically nonlinear activation function, \f$ x
 * \f$ is the input of the layer and \f$ W \f$ is a weight matrix.
 *
 * Neural networks with one fully connected hidden layer and a nonlinear
 * activation function are universal function approximators, i. e. with a
 * sufficient number of nodes any function can be approximated with arbitrary
 * precision. However, in practice the number of nodes could be very large and
 * overfitting is a problem. Therefore it is sometimes better to add more
 * hidden layers. Note that this could cause another problem: the gradients
 * vanish in the lower layers such that these cannot be trained properly. If
 * you want to apply a complex neural network to tasks like image recognition
 * you could instead try Convolutional layers and pooling layers (MaxPooling,
 * Subsampling) in the lower layers. These can be trained surprisingly well in
 * deep architectures.
 *
 * [1] Kurt Hornik, Maxwell B. Stinchcombe and Halbert White:
 * Multilayer feedforward networks are universal approximators,
 * Neural Networks 2 (5), pp. 359-366, 1989.
 */
class FullyConnected : public Layer
{
  int I, J;
  bool bias;
  ActivationFunction act;
  fpt stdDev;
  fpt dropoutProbability;
  Mt W;
  Mt Wd;
  Vt* x;
  Vt a;
  Vt y;
  Vt yd;
  Vt deltas;
  Vt e;

public:
  FullyConnected(OutputInfo info, int J, bool bias, ActivationFunction act,
                 fpt stdDev, fpt dropoutProbability);
  virtual OutputInfo initialize(std::vector<fpt*>& parameterPointers,
                                std::vector<fpt*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters() {}
  virtual void forwardPropagate(Vt* x, Vt*& y, bool dropout);
  virtual void backpropagate(Vt* ein, Vt*& eout);
};

}
