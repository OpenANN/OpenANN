#pragma once

#include <layers/Layer.h>
#include <io/Logger.h>
#include <ActivationFunctions.h>

namespace OpenANN {

class Subsampling : public Layer
{
  Logger debugLogger;
  int I, fm, inRows, inCols, kernelRows, kernelCols;
  bool bias, weightForBias;
  ActivationFunction act;
  fpt stdDev;
  Vt* x;
  //! feature maps X output rows X output cols
  std::vector<Mt> W;
  std::vector<Mt> Wd;
  //! feature maps X output rows X output cols
  std::vector<Mt> Wb;
  std::vector<Mt> Wbd;
  Vt a;
  Vt y;
  Vt yd;
  Vt deltas;
  Vt e;
  int fmInSize, outRows, outCols, fmOutSize, maxRow, maxCol;

public:
  Subsampling(OutputInfo info, int kernelRows, int kernelCols, bool bias,
              ActivationFunction act, fpt stdDev);
  virtual OutputInfo initialize(std::vector<fpt*>& parameterPointers, std::vector<fpt*>& parameterDerivativePointers);
  virtual void forwardPropagate(Vt* x, Vt*& y);
  virtual void backpropagate(Vt* ein, Vt*& eout);
};

}
