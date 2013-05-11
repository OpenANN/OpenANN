#include <OpenANN/layers/Convolutional.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/OpenANNException.h>

namespace OpenANN {

Convolutional::Convolutional(OutputInfo info, int featureMaps, int kernelRows,
                             int kernelCols, bool bias, ActivationFunction act,
                             double stdDev)
  : I(info.outputs()), fmin(info.dimensions[0]), inRows(info.dimensions[1]),
    inCols(info.dimensions[2]), fmout(featureMaps), kernelRows(kernelRows),
    kernelCols(kernelCols), bias(bias), act(act),
    stdDev(stdDev), x(0), e(1, I)
{
}

OutputInfo Convolutional::initialize(std::vector<double*>& parameterPointers,
                                     std::vector<double*>& parameterDerivativePointers)
{
  OutputInfo info;
  info.dimensions.push_back(fmout);
  outRows = inRows-kernelRows/2*2;
  outCols = inCols-kernelCols/2*2;
  fmOutSize = outRows * outCols;
  info.dimensions.push_back(outRows);
  info.dimensions.push_back(outCols);
  fmInSize = inRows * inCols;
  maxRow = inRows-kernelRows+1;
  maxCol = inCols-kernelCols+1;

  W.resize(fmout, std::vector<Eigen::MatrixXd>(fmin, Eigen::MatrixXd(kernelRows, kernelCols)));
  Wd.resize(fmout, std::vector<Eigen::MatrixXd>(fmin, Eigen::MatrixXd(kernelRows, kernelCols)));
  int numParams = fmout*kernelRows*kernelCols;
  if(bias)
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
      if(bias)
      {
        parameterPointers.push_back(&Wb(fmo, fmi));
        parameterDerivativePointers.push_back(&Wbd(fmo, fmi));
      }
    }
  }

  initializeParameters();

  a.resize(1, info.outputs());
  y.resize(1, info.outputs());
  yd.resize(1, info.outputs());
  deltas.resize(1, info.outputs());

  if(info.outputs() < 1)
    throw OpenANNException("Number of outputs in convolutional layer is below"
                           " 1. You should either choose a smaller filter"
                           " size or generate a bigger input.");
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
          W[fmo][fmi](kr, kc) = rng.sampleNormalDistribution<double>() * stdDev;
      if(bias)
        Wb(fmo, fmi) = rng.sampleNormalDistribution<double>() * stdDev;
    }
  }
}

void Convolutional::forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y, bool dropout)
{
  this->x = x;

  OPENANN_CHECK_EQUALS(x->cols(), fmin * inRows * inRows);
  OPENANN_CHECK_EQUALS(this->y.cols(), fmout * outRows * outCols);

  a.fill(0.0);
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
              OPENANN_CHECK(outputIdx < a.cols());
              OPENANN_CHECK(inputIdx < x->cols());
              a(0, outputIdx) += W[fmo][fmi](kr, kc)*(*x)(0, inputIdx);
            }
          }
          if(bias && fmi == 0)
            a(0, outputIdx) += Wb(fmo, fmi);
        }
      }
    }
  }

  activationFunction(act, a, this->y);

  y = &(this->y);
}

void Convolutional::backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout)
{
  // Derive activations
  activationFunctionDerivative(act, y, yd);
  deltas = yd.cwiseProduct(*ein);

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
              OPENANN_CHECK(outputIdx < deltas.cols());
              OPENANN_CHECK(inputIdx < x->cols());
              e(inputIdx) += W[fmo][fmi](kr, kc)*deltas(0, outputIdx);
              Wd[fmo][fmi](kr, kc) += deltas(0, outputIdx) * (*x)(0, inputIdx);
            }
          }
          if(bias && fmi == 0)
            Wbd(fmo, fmi) += deltas(0, outputIdx);
        }
      }
    }
  }

  eout = &e;
}

Eigen::MatrixXd& Convolutional::getOutput()
{
  return y;
}

}
