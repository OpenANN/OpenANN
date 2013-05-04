#pragma once

#include <OpenANN/layers/Layer.h>
#include <OpenANN/ActivationFunctions.h>

namespace OpenANN {

/**
 * @class Input
 *
 * Input layer.
 */
class Input : public Layer
{
  int J, dim1, dim2, dim3;
  Eigen::VectorXd* x;

public:
  Input(int dim1, int dim2, int dim3);
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters() {}
  virtual void forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout);
  virtual void backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout);
  virtual Eigen::VectorXd& getOutput();
};

}
