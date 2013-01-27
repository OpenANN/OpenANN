#pragma once

#include <layers/Layer.h>
#include <io/Logger.h>
#include <ActivationFunctions.h>

namespace OpenANN {

class FullyConnected : public Layer
{
  Logger debugLogger;
  int I, J;
  bool bias;
  ActivationFunction act;
  fpt stdDev;
  Mt W;
  Mt Wd;
  Vt* x;
  Vt a;
  Vt y;
  Vt yd;
  Vt deltas;
  Vt e;

public:
  FullyConnected(OutputInfo info, int J, bool bias, ActivationFunction act, fpt stdDev);
  virtual OutputInfo initialize(std::list<fpt*>& parameterPointers, std::list<fpt*>& parameterDerivativePointers);
  virtual void forwardPropagate(Vt* x, Vt*& y);
  virtual void backpropagate(Vt* ein, Vt*& eout);
};

}
