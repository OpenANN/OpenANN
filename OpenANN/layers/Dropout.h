#pragma once

#include <OpenANN/layers/Layer.h>

namespace OpenANN {

/**
 * @class Dropout
 *
 * Dropout mask.
 *
 * The dropout technique tries to minimize similarities of neurons in one
 * layer by randomly suppressing the output of neurons during training [1].
 * After training, we use the "mean network", i.e. we reduce the output of
 * each neuron to compensate that all will be active.
 *
 * [1] Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I. and
 * Salakhutdinov, R. R.:
 * Improving neural networks by preventing co-adaptation of feature detectors,
 * 2012.
 */
class Dropout : public Layer
{
  int I;
  bool bias;
  double dropoutProbability;
  Eigen::VectorXd dropoutMask;
  Eigen::VectorXd y;
  Eigen::VectorXd e;
public:
  Dropout(OutputInfo info, double dropoutProbability);
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters() {}
  virtual void updatedParameters() {}
  virtual void forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout);
  virtual void backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout);
  virtual Eigen::VectorXd& getOutput();
};

}
