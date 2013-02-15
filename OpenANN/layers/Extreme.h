#pragma once

#include <layers/Layer.h>
#include <ActivationFunctions.h>

namespace OpenANN {

class Extreme : public Layer
{
  int I, J;
  bool bias;
  ActivationFunction act;
  fpt stdDev;
  Mt W;
  Vt* x;
  Vt a;
  Vt y;
  Vt yd;
  Vt deltas;
  Vt e;

public:
  Extreme(OutputInfo info, int J, bool bias, ActivationFunction act,
          fpt stdDev);
  virtual OutputInfo initialize(std::vector<fpt*>& parameterPointers,
                                std::vector<fpt*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters() {}
  virtual void forwardPropagate(Vt* x, Vt*& y, bool dropout);
  virtual void backpropagate(Vt* ein, Vt*& eout);
};

}
