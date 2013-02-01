#include <layers/Input.h>

namespace OpenANN {

Input::Input(int dim1, int dim2, int dim3, bool bias)
  : J(dim1*dim2*dim3), dim1(dim1), dim2(dim2), dim3(dim3), bias(bias),
    y(J+bias)
{
}

OutputInfo Input::initialize(std::vector<fpt*>& parameterPointers,
                             std::vector<fpt*>& parameterDerivativePointers)
{
  // Bias component will not change after initialization
  if(bias)
    y(J) = fpt(1.0);
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

void Input::forwardPropagate(Vt* x, Vt*& y)
{
  // Copy entries and add bias
  for(int i = 0; i < J; i++)
    this->y(i) = (*x)(i);
  y = &(this->y);
}

void Input::backpropagate(Vt* ein, Vt*& eout)
{
  // Do nothing.
}

}
