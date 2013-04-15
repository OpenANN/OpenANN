#include <OpenANN/layers/MaxPooling.h>
#include <OpenANN/util/AssertionMacros.h>
#include <limits>
#include <algorithm>

namespace OpenANN {

MaxPooling::MaxPooling(OutputInfo info, int kernelRows, int kernelCols, bool bias)
  : I(info.outputs()), fm(info.dimensions[0]),
    inRows(info.dimensions[1]), inCols(info.dimensions[2]),
    kernelRows(kernelRows), kernelCols(kernelCols), bias(bias), x(0), e(I)
{
}

OutputInfo MaxPooling::initialize(std::vector<fpt*>& parameterPointers,
                                   std::vector<fpt*>& parameterDerivativePointers)
{
  OutputInfo info;
  info.bias = bias;
  info.dimensions.push_back(fm);
  outRows = inRows/kernelRows;
  outCols = inCols/kernelCols;
  fmOutSize = outRows * outCols;
  info.dimensions.push_back(outRows);
  info.dimensions.push_back(outCols);
  fmInSize = inRows * inCols;
  maxRow = inRows-kernelRows+1;
  maxCol = inCols-kernelCols+1;

  y.resize(info.outputs());
  if(bias)
    y(y.rows()-1) = 1.0;
  deltas.resize(info.outputs()-bias);

  return info;
}

void MaxPooling::initializeParameters()
{
}

void MaxPooling::forwardPropagate(Vt* x, Vt*& y, bool dropout)
{
  this->x = x;

  OPENANN_CHECK(x->rows() == fm * inRows * inRows
      || x->rows() == fm * inRows * inRows + 1);
  OPENANN_CHECK_EQUALS(this->y.rows(), fm * outRows * outCols + bias);

  int outputIdx = 0;
  int inputIdx = 0;
  for(int fmo = 0; fmo < fm; fmo++)
  {
    for(int ri = 0, ro = 0; ri < maxRow; ri+=kernelRows, ro++)
    {
      int rowBase = fmo*fmInSize + ri*inCols;
      for(int ci = 0, co = 0; ci < maxCol; ci+=kernelCols, co++, outputIdx++)
      {
        fpt m = -std::numeric_limits<fpt>::max();
        for(int kr = 0; kr < kernelRows; kr++)
        {
          inputIdx = rowBase + ci;
          for(int kc = 0; kc < kernelCols; kc++, inputIdx++)
            m = std::max(m, (*x)(inputIdx));
        }
        this->y(outputIdx) = m;
      }
    }
  }

  y = &(this->y);
}

void MaxPooling::backpropagate(Vt* ein, Vt*& eout)
{
  for(int j = 0; j < deltas.rows(); j++)
    deltas(j) = (*ein)(j);

  e.fill(0.0);
  int outputIdx = 0;
  int inputIdx = 0;
  for(int fmo = 0; fmo < fm; fmo++)
  {
    for(int ri = 0, ro = 0; ri < maxRow; ri+=kernelRows, ro++)
    {
      int rowBase = fmo*fmInSize + ri*inCols;
      for(int ci = 0, co = 0; ci < maxCol; ci+=kernelCols, co++, outputIdx++)
      {
        fpt m = -std::numeric_limits<fpt>::max();
        int idx = -1;
        for(int kr = 0; kr < kernelRows; kr++)
        {
          inputIdx = rowBase + ci;
          for(int kc = 0; kc < kernelCols; kc++, inputIdx++)
            if((*x)(inputIdx) > m)
            {
              m = (*x)(inputIdx);
              idx = inputIdx;
            }
        }
        e(idx) = deltas(outputIdx);
      }
    }
  }

  eout = &e;
}

Vt& MaxPooling::getOutput()
{
  return y;
}

}
