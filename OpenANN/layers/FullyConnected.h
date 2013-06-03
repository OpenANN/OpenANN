#ifndef OPENANN_LAYERS_FULLY_CONNECTED_H_
#define OPENANN_LAYERS_FULLY_CONNECTED_H_

#include <OpenANN/layers/Layer.h>
#include <OpenANN/ActivationFunctions.h>

namespace OpenANN
{

/**
 * @class FullyConnected
 *
 * Fully connected layer.
 *
 * Each neuron in the previous layer is taken as input for each neuron of this
 * layer. Forward propagation is usually done by \f$ a = W \cdot x + b, y =
 * g(a) \f$, where \f$ a \f$ is the activation vector, \f$ y \f$ is the
 * output, \f$ g \f$ a typically nonlinear activation function that operates
 * on a vector, \f$ x \f$ is the input of the layer, \f$ W \f$ is a weight
 * matrix and \f$ b \f$ is a bias vector.
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
  double stdDev;
  double maxSquaredWeightNorm;
  Eigen::MatrixXd W;
  Eigen::MatrixXd Wd;
  Eigen::VectorXd b;
  Eigen::VectorXd bd;
  Eigen::MatrixXd* x;
  Eigen::MatrixXd a;
  Eigen::MatrixXd y;
  Eigen::MatrixXd yd;
  Eigen::MatrixXd deltas;
  Eigen::MatrixXd e;

public:
  FullyConnected(OutputInfo info, int J, bool bias, ActivationFunction act,
                 double stdDev, double maxSquaredWeightNorm);
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters();
  virtual void forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y, bool dropout);
  virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout);
  virtual Eigen::MatrixXd& getOutput();
};

}

#endif // OPENANN_LAYERS_FULLY_CONNECTED_H_
