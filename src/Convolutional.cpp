#include <layers/Convolutional.h>
#include <Random.h>
#include <AssertionMacros.h>

namespace OpenANN {

Convolutional::Convolutional(OutputInfo info, int featureMaps, int kernelRows,
                             int kernelCols, bool bias, ActivationFunction act,
                             fpt stdDev)
  : I(info.outputs()), fmin(info.dimensions[0]), inRows(info.dimensions[1]),
    inCols(info.dimensions[2]), fmout(featureMaps), kernelRows(kernelRows),
    kernelCols(kernelCols), bias(bias), weightForBias(info.bias), act(act),
    stdDev(stdDev), x(0), e(I)
{
}

OutputInfo Convolutional::initialize(std::vector<fpt*>& parameterPointers,
                                     std::vector<fpt*>& parameterDerivativePointers)
{
  OutputInfo info;
  info.bias = bias;
  info.dimensions.push_back(fmout);
  outRows = inRows-kernelRows/2*2;
  outCols = inCols-kernelCols/2*2;
  fmOutSize = outRows * outCols;
  info.dimensions.push_back(outRows);
  info.dimensions.push_back(outCols);
  fmInSize = inRows * inCols;
  maxRow = inRows-kernelRows+1;
  maxCol = inCols-kernelCols+1;

  W.resize(fmout, std::vector<Mt>(fmin, Mt(kernelRows, kernelCols)));
  Wd.resize(fmout, std::vector<Mt>(fmin, Mt(kernelRows, kernelCols)));
  int numParams = fmout*kernelRows*kernelCols;
  if(weightForBias)
  {
    Wb.resize(fmout, fmin);
    Wbd.resize(fmout, fmin);
    numParams += fmout * fmin;
  }
  parameterPointers.reserve(parameterPointers.size() + numParams);
  parameterDerivativePointers.reserve(parameterDerivativePointers.size() + numParams);
  for(int fmo = 0; fmo < fmout; fmo++)
  {
    for(int fmi = 0; fmi < fmin; fmi++)
    {
      for(int kr = 0; kr < kernelRows; kr++)
      {
        for(int kc = 0; kc < kernelCols; kc++)
        {
          parameterPointers.push_back(&W[fmo][fmi](kr, kc));
          parameterDerivativePointers.push_back(&Wd[fmo][fmi](kr, kc));
        }
      }
      if(weightForBias)
      {
        parameterPointers.push_back(&Wb(fmo, fmi));
        parameterDerivativePointers.push_back(&Wbd(fmo, fmi));
      }
    }
  }

  initializeParameters();

  a.resize(info.outputs()-bias);
  y.resize(info.outputs());
  if(bias)
    y(y.rows()-1) = 1.0;
  yd.resize(info.outputs()-bias);
  deltas.resize(info.outputs()-bias);

  return info;
}

void Convolutional::initializeParameters()
{
  RandomNumberGenerator rng;
  for(int fmo = 0; fmo < fmout; fmo++)
  {
    for(int fmi = 0; fmi < fmin; fmi++)
    {
      for(int kr = 0; kr < kernelRows; kr++)
        for(int kc = 0; kc < kernelCols; kc++)
          W[fmo][fmi](kr, kc) = rng.sampleNormalDistribution<fpt>() * stdDev;
      if(weightForBias)
        Wb(fmo, fmi) = rng.sampleNormalDistribution<fpt>() * stdDev;
    }
  }
}

void Convolutional::forwardPropagate(Vt* x, Vt*& y, bool dropout)
{
  this->x = x;

  OPENANN_CHECK_EQUALS(x->rows(), fmin * inRows * inRows + weightForBias);
  OPENANN_CHECK_EQUALS(this->y.rows(), fmout * outRows * outCols + bias);

  this->a.fill(0.0);
  for(int fmo = 0; fmo < fmout; fmo++)
  {
    int fmInBase = 0;
    for(int fmi = 0; fmi < fmin; fmi++, fmInBase+=fmInSize)
    {
      int outputIdx = fmo * fmOutSize;
      for(int row = 0; row < maxRow; row++)
      {
        for(int col = 0; col < maxCol; col++, outputIdx++)
        {
          int rowBase = fmInBase+row*inCols;
          for(int kr = 0, kri = row; kr < kernelRows; kr++, kri++, rowBase+=inCols)
          {
            int inputIdx = rowBase+col;
            for(int kc = 0, kci = col; kc < kernelCols; kc++, kci++, inputIdx++)
            {
              OPENANN_CHECK(outputIdx < a.rows());
              OPENANN_CHECK(inputIdx < x->rows());
              a(outputIdx) += W[fmo][fmi](kr, kc)*(*x)(inputIdx);
            }
          }
        }
      }
      if(weightForBias)
        a(outputIdx-1) += Wb(fmo, fmi);
    }
  }

  activationFunction(act, a, this->y);

  y = &(this->y);
}

void Convolutional::backpropagate(Vt* ein, Vt*& eout)
{
  // Derive activations
  activationFunctionDerivative(act, y, yd);
  for(int j = 0; j < deltas.rows(); j++)
    deltas(j) = yd(j) * (*ein)(j);

  e.fill(0.0);
  Wbd.fill(0.0);
  for(int fmo = 0; fmo < fmout; fmo++)
  {
    int fmInBase = 0;
    for(int fmi = 0; fmi < fmin; fmi++, fmInBase+=fmInSize)
    {
      Wd[fmo][fmi].fill(0.0);
      int outputIdx = fmo * fmOutSize;
      for(int row = 0; row < maxRow; row++)
      {
        for(int col = 0; col < maxCol; col++, outputIdx++)
        {
          int rowBase = fmInBase+row*inCols;
          for(int kr = 0, kri = row; kr < kernelRows; kr++, kri++, rowBase+=inCols)
          {
            int inputIdx = rowBase+col;
            for(int kc = 0, kci = col; kc < kernelCols; kc++, kci++, inputIdx++)
            {
              OPENANN_CHECK(outputIdx < a.rows());
              OPENANN_CHECK(inputIdx < x->rows());
              e(inputIdx) += W[fmo][fmi](kr, kc)*deltas(outputIdx);
              Wd[fmo][fmi](kr, kc) += deltas(outputIdx) * (*x)(inputIdx);
            }
          }
        }
      }
      if(weightForBias)
        Wbd(fmo, fmi) += deltas(outputIdx-1);
    }
  }

  eout = &e;
}

}
