#pragma once

#include <layers/Layer.h>
#include <ActivationFunctions.h>

namespace OpenANN {

class Compressed : public Layer
{
  int I, J, M;
  bool bias;
  ActivationFunction act;
  fpt stdDev;
  fpt dropoutProbability;
  Mt W;
  Mt Wd;
  Mt phi;
  Mt alpha;
  Mt alphad;
  Vt* x;
  Vt a;
  Vt y;
  Vt yd;
  Vt deltas;
  Vt e;

public:
  Compressed(OutputInfo info, int J, int M, bool bias, ActivationFunction act,
             const std::string& compression, fpt stdDev,
             fpt dropoutProbability);
  virtual OutputInfo initialize(std::vector<fpt*>& parameterPointers,
                                std::vector<fpt*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters();
  virtual void forwardPropagate(Vt* x, Vt*& y, bool dropout);
  virtual void backpropagate(Vt* ein, Vt*& eout);
};

}
