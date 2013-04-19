#pragma once

#include <OpenANN/layers/Layer.h>
#include <OpenANN/ActivationFunctions.h>

namespace OpenANN {

/**
 * @class Extreme
 *
 * Fully connected layer with fixed random weights.
 *
 * This kind of layer is used in Extreme Learning Machines (ELMs). It projects
 * low dimensional data onto a higher dimensional space such that a linear
 * classifier or regression algorithm in the next layer could approximate
 * arbitrary functions. The advantage of this concept is that a closed form
 * solution is available. The disadvantage is that the number of required
 * nodes is usually much larger than in conventional multilayer neural
 * networks.
 *
 * [1] Guang-Bin Huang, Qin-Yu Zhu and Chee-Kheong Siew:
 * Extreme learning machine: Theory and applications,
 * Neurocomputing 70 (1–3), pp. 489-501, 2006.
 */
class Extreme : public Layer
{
  int I, J;
  bool bias;
  ActivationFunction act;
  double stdDev;
  Eigen::MatrixXd W;
  Eigen::VectorXd* x;
  Eigen::VectorXd a;
  Eigen::VectorXd y;
  Eigen::VectorXd yd;
  Eigen::VectorXd deltas;
  Eigen::VectorXd e;

public:
  Extreme(OutputInfo info, int J, bool bias, ActivationFunction act,
          double stdDev);
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters() {}
  virtual void forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout);
  virtual void backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout);
  virtual Eigen::VectorXd& getOutput();
};

}
