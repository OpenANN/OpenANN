#pragma once

#include <OpenANN/layers/Layer.h>

namespace OpenANN {

/**
 * @class LocalResponseNormalization
 *
 * Local response normalization.
 *
 * This layer encourages the competition between neurons at the same spatial
 * positions. It implements a form of lateral inhibition that is found in
 * real neurons. It can be interpreted as "brightness normalization" in visual
 * models [1].
 *
 * It requires a three-dimensional input so that we can calculate the output
 * as
 *
 * \f$ y^i_{rc} = x^i_{rc} / \left( k +
 *     \alpha \sum_{j=max(0, i-n/2)}^{min(N-1, i+n/2)}
 *     x^j_{rc} \right)^{\beta}, \f$
 *
 * where i is the index of the feature map (or filter bank), r is the row, and
 * c the column of the neuron respectively, N is the number of feature maps
 * and \f$ k, n, \alpha, \beta \f$ are hyperparameters that have to be found
 * with a validation set. A reasonable choice is e.g. \f$ k=2, n=5,
 * \alpha=10^{-4}, \beta=0.75 \f$ [1].
 *
 * [1] Krizhevsky, Alexander, Sutskever, Ilya and Hinton, Geoffrey E.:
 * ImageNet Classification with Deep Convolutional Neural Networks,
 * Advances in Neural Information Processing Systems 25, pp. 1106–1114, 2012.
 *
 * [2] Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I. and
 * Salakhutdinov, R. R.:
 * Improving neural networks by preventing co-adaptation of feature detectors,
 * 2012.
 */
class LocalResponseNormalization : public Layer
{
  int I, fm, rows, cols;
  int fmSize;
  bool bias;
  double k;
  int n;
  double alpha;
  double beta;
  Eigen::VectorXd* x;
  Eigen::VectorXd denoms;
  Eigen::VectorXd y;
  Eigen::VectorXd etmp;
  Eigen::VectorXd e;

public:
  LocalResponseNormalization(OutputInfo info, bool bias, double k, int n,
                             double alpha, double beta);
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters() {}
  virtual void updatedParameters() {}
  virtual void forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout);
  virtual void backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout);
  virtual Eigen::VectorXd& getOutput();
};

}
