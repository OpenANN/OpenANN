#pragma once

#include <OpenANN/layers/Layer.h>
#include <OpenANN/io/Logger.h>
#include <OpenANN/ActivationFunctions.h>

namespace OpenANN {

class AlphaBetaFilter : public Layer
{
  int I, J;
  fpt deltaT;
  bool bias;
  fpt stdDev;
  Vt gamma;
  Vt gammad;
  Vt alpha;
  Vt beta;
  bool first;
  Vt* x;
  Vt y;

public:
  AlphaBetaFilter(OutputInfo info, fpt deltaT, bool bias, fpt stdDev);
  virtual OutputInfo initialize(std::vector<fpt*>& parameterPointers, std::vector<fpt*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters();
  virtual void reset();
  virtual void forwardPropagate(Vt* x, Vt*& y, bool dropout);
  virtual void backpropagate(Vt* ein, Vt*& eout);
  virtual Vt& getOutput();
};

}
