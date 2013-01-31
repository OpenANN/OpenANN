#pragma once

#include <layers/Layer.h>
#include <ActivationFunctions.h>

namespace OpenANN {

class Input : public Layer
{
  int J, dim1, dim2, dim3;
  bool bias;
  Vt y;

public:
  Input(int dim1, int dim2, int dim3, bool bias);
  virtual OutputInfo initialize(std::vector<fpt*>& parameterPointers, std::vector<fpt*>& parameterDerivativePointers);
  virtual void forwardPropagate(Vt* x, Vt*& y);
  virtual void backpropagate(Vt* ein, Vt*& eout);
};

}
