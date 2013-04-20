#include <OpenANN/layers/Input.h>
#include <OpenANN/util/Random.h>

namespace OpenANN {

Input::Input(int dim1, int dim2, int dim3, bool bias)
  : J(dim1*dim2*dim3), dim1(dim1), dim2(dim2), dim3(dim3), bias(bias), y(J+bias)
{
}

OutputInfo Input::initialize(std::vector<double*>& parameterPointers,
                             std::vector<double*>& parameterDerivativePointers)
{
  // Bias component will not change after initialization
  if(bias)
    y(J) = 1.0;
  OutputInfo info;
  info.bias = bias;
  info.dimensions.push_back(dim1);
  info.dimensions.push_back(dim2);
  info.dimensions.push_back(dim3);
  return info;
}

void Input::initializeParameters()
{
  // Do nothing.
}

void Input::forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout)
{
  // Copy entries and add bias
  for(int i = 0; i < J; i++)
    this->y(i) = (*x)(i);
  y = &(this->y);
}

void Input::backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout)
{
  // Do nothing.
}

Eigen::VectorXd& Input::getOutput()
{
  return y;
}

}
