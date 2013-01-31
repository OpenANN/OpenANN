#include <layers/Subsampling.h>
#include <Random.h>
#include <AssertionMacros.h>

namespace OpenANN {

Subsampling::Subsampling(OutputInfo info, int kernelRows, int kernelCols,
                         bool bias, ActivationFunction act, fpt stdDev)
  : debugLogger(Logger::CONSOLE), I(info.outputs()), fm(info.dimensions[0]),
    inRows(info.dimensions[1]), inCols(info.dimensions[2]),
    kernelRows(kernelRows), kernelCols(kernelCols), bias(bias),
    weightForBias(info.bias), act(act), stdDev(stdDev), x(0), e(I)
{
}

OutputInfo Subsampling::initialize(std::vector<fpt*>& parameterPointers,
                                   std::vector<fpt*>& parameterDerivativePointers)
{
  y.resize(0);
  RandomNumberGenerator rng;
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

  W.resize(fm, Mt(outRows, outCols));
  Wd.resize(fm, Mt(outRows, outCols));
  if(weightForBias)
  {
    Wb.resize(fm, Mt(outRows, outCols));
    Wbd.resize(fm, Mt(outRows, outCols));
  }
  for(int fmo = 0; fmo < fm; fmo++)
  {
    for(int r = 0; r < outRows; r++)
    {
      for(int c = 0; c < outCols; c++)
      {
        W[fmo](r, c) = rng.sampleNormalDistribution<fpt>() * stdDev;
        parameterPointers.push_back(&W[fmo](r, c));
        parameterDerivativePointers.push_back(&Wd[fmo](r, c));
        if(weightForBias)
        {
          Wb[fmo](r, c) = rng.sampleNormalDistribution<fpt>() * stdDev;
          parameterPointers.push_back(&Wb[fmo](r, c));
          parameterDerivativePointers.push_back(&Wbd[fmo](r, c));
        }
      }
    }
  }
  a.resize(info.outputs()-bias);
  y.resize(info.outputs());
  if(bias)
    y(y.rows()-1) = 1.0;
  yd.resize(info.outputs()-bias);
  deltas.resize(info.outputs()-bias);

  return info;
}

void Subsampling::forwardPropagate(Vt* x, Vt*& y)
{
  this->x = x;

  OPENANN_CHECK_EQUALS(x->rows(), fm * inRows * inRows + weightForBias);
  OPENANN_CHECK_EQUALS(this->y.rows(), fm * outRows * outCols + bias);

  a.fill(0.0);
  int outputIdx = 0;
  int inputIdx = 0;
  for(int fmo = 0; fmo < fm; fmo++)
  {
    for(int ri = 0, ro = 0; ri < maxRow; ri+=kernelRows, ro++)
    {
      for(int ci = 0, co = 0; ci < maxCol; ci+=kernelCols, co++, outputIdx++)
      {
        for(int kr = 0; kr < kernelRows; kr++)
          for(int kc = 0; kc < kernelCols; kc++, inputIdx++)
            a(outputIdx) += (*x)(inputIdx) * W[fmo](ro, co);
        if(weightForBias)
          a(outputIdx) += Wb[fmo](ro, co);
      }
    }
  }

  activationFunction(act, a, this->y);

  y = &(this->y);
}

void Subsampling::backpropagate(Vt* ein, Vt*& eout)
{
  // Derive activations
  activationFunctionDerivative(act, y, yd);
  for(int j = 0; j < deltas.rows(); j++)
    deltas(j) = yd(j) * (*ein)(j);

  e.fill(0.0);
  int outputIdx = 0;
  int inputIdx = 0;
  for(int fmo = 0; fmo < fm; fmo++)
  {
    Wd[fmo].fill(0.0);
    if(weightForBias)
      Wbd[fmo].fill(0.0);
    for(int ri = 0, ro = 0; ri < maxRow; ri+=kernelRows, ro++)
    {
      for(int ci = 0, co = 0; ci < maxCol; ci+=kernelCols, co++, outputIdx++)
      {
        for(int kr = 0; kr < kernelRows; kr++)
          for(int kc = 0; kc < kernelCols; kc++, inputIdx++)
          {
            e(inputIdx) += W[fmo](ro, co)*deltas(outputIdx);
            Wd[fmo](ro, co) += deltas(outputIdx) * (*x)(inputIdx);
          }
        if(weightForBias)
          Wbd[fmo](ro, co) += deltas(outputIdx);
      }
    }
  }

  eout = &e;
}

}
