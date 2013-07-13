#include <OpenANN/layers/Subsampling.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/OpenANNException.h>

namespace OpenANN
{

Subsampling::Subsampling(OutputInfo info, int kernelRows, int kernelCols,
                         bool bias, ActivationFunction act, double stdDev,
                         Regularization regularization)
  : I(info.outputs()), fm(info.dimensions[0]), inRows(info.dimensions[1]),
    inCols(info.dimensions[2]), kernelRows(kernelRows),
    kernelCols(kernelCols), bias(bias), act(act), stdDev(stdDev), x(0),
    e(1, I), fmInSize(-1), outRows(-1), outCols(-1), fmOutSize(-1),
    maxRow(-1), maxCol(-1), regularization(regularization)
{
}

OutputInfo Subsampling::initialize(std::vector<double*>& parameterPointers,
                                   std::vector<double*>& parameterDerivativePointers)
{
  OutputInfo info;
  info.dimensions.push_back(fm);
  outRows = inRows / kernelRows;
  outCols = inCols / kernelCols;
  fmOutSize = outRows * outCols;
  info.dimensions.push_back(outRows);
  info.dimensions.push_back(outCols);
  fmInSize = inRows * inCols;
  maxRow = inRows - kernelRows + 1;
  maxCol = inCols - kernelCols + 1;

  W.resize(fm, Eigen::MatrixXd(outRows, outCols));
  Wd.resize(fm, Eigen::MatrixXd(outRows, outCols));
  int numParams = fm * outRows * outCols * kernelRows * kernelCols;
  if(bias)
  {
    Wb.resize(fm, Eigen::MatrixXd(outRows, outCols));
    Wbd.resize(fm, Eigen::MatrixXd(outRows, outCols));
    numParams += fm * outRows * outCols;
  }
  parameterPointers.reserve(parameterPointers.size() + numParams);
  parameterDerivativePointers.reserve(parameterDerivativePointers.size() + numParams);
  for(int fmo = 0; fmo < fm; fmo++)
  {
    for(int r = 0; r < outRows; r++)
    {
      for(int c = 0; c < outCols; c++)
      {
        parameterPointers.push_back(&W[fmo](r, c));
        parameterDerivativePointers.push_back(&Wd[fmo](r, c));
        if(bias)
        {
          parameterPointers.push_back(&Wb[fmo](r, c));
          parameterDerivativePointers.push_back(&Wbd[fmo](r, c));
        }
      }
    }
  }

  initializeParameters();

  a.resize(1, info.outputs());
  y.resize(1, info.outputs());
  yd.resize(1, info.outputs());
  deltas.resize(1, info.outputs());

  if(info.outputs() < 1)
    throw OpenANNException("Number of outputs in subsampling layer is below"
                           " 1. You should either choose a smaller filter"
                           " size or generate a bigger input.");
  OPENANN_CHECK(fmInSize > 0);
  OPENANN_CHECK(outRows > 0);
  OPENANN_CHECK(outCols > 0);
  OPENANN_CHECK(fmOutSize > 0);
  OPENANN_CHECK(maxRow > 0);
  OPENANN_CHECK(maxCol > 0);

  return info;
}

void Subsampling::initializeParameters()
{
  RandomNumberGenerator rng;
  for(int fmo = 0; fmo < fm; fmo++)
  {
    for(int r = 0; r < outRows; r++)
    {
      for(int c = 0; c < outCols; c++)
      {
        W[fmo](r, c) = rng.sampleNormalDistribution<double>() * stdDev;
        if(bias)
          Wb[fmo](r, c) = rng.sampleNormalDistribution<double>() * stdDev;
      }
    }
  }
}

void Subsampling::forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                                   bool dropout)
{
  const int N = x->rows();
  this->a.conservativeResize(N, Eigen::NoChange);
  this->y.conservativeResize(N, Eigen::NoChange);
  this->x = x;

  OPENANN_CHECK_EQUALS(x->cols(), fm * inRows * inCols);
  OPENANN_CHECK_EQUALS(this->y.cols(), fm * outRows * outCols);

  a.setZero();
  #pragma omp parallel for
  for(int n = 0; n < N; n++)
  {
    int outputIdx = 0;
    int inputIdx = 0;
    for(int fmo = 0; fmo < fm; fmo++)
    {
      for(int ri = 0, ro = 0; ri < maxRow; ri += kernelRows, ro++)
      {
        int rowBase = fmo * fmInSize + ri * inCols;
        for(int ci = 0, co = 0; ci < maxCol; ci += kernelCols, co++, outputIdx++)
        {
          for(int kr = 0; kr < kernelRows; kr++)
          {
            inputIdx = rowBase + ci;
            for(int kc = 0; kc < kernelCols; kc++, inputIdx++)
              a(n, outputIdx) += (*x)(n, inputIdx) * W[fmo](ro, co);
          }
          if(bias)
            a(n, outputIdx) += Wb[fmo](ro, co);
        }
      }
    }
  }

  activationFunction(act, a, this->y);

  y = &(this->y);
}

void Subsampling::backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,
                                bool backpropToPrevious)
{
  const int N = a.rows();
  yd.conservativeResize(N, Eigen::NoChange);
  e.conservativeResize(N, Eigen::NoChange);
  // Derive activations
  activationFunctionDerivative(act, y, yd);
  deltas = yd.cwiseProduct(*ein);

  e.setZero();
  for(int fmo = 0; fmo < fm; fmo++)
  {
    Wd[fmo].setZero();
    if(bias)
      Wbd[fmo].setZero();
  }
  for(int n = 0; n < N; n++)
  {
    int outputIdx = 0;
    int inputIdx = 0;
    for(int fmo = 0; fmo < fm; fmo++)
    {
      for(int ri = 0, ro = 0; ri < maxRow; ri += kernelRows, ro++)
      {
        int rowBase = fmo * fmInSize + ri * inCols;
        for(int ci = 0, co = 0; ci < maxCol; ci += kernelCols, co++, outputIdx++)
        {
          for(int kr = 0; kr < kernelRows; kr++)
          {
            inputIdx = rowBase + ci;
            for(int kc = 0; kc < kernelCols; kc++, inputIdx++)
            {
              e(n, inputIdx) += W[fmo](ro, co) * deltas(n, outputIdx);
              Wd[fmo](ro, co) += deltas(n, outputIdx) * (*x)(n, inputIdx);
            }
          }
          if(bias)
            Wbd[fmo](ro, co) += deltas(n, outputIdx);
        }
      }
    }
  }

  if(regularization.l1Penalty > 0.0)
  {
    for(int fmo = 0; fmo < fm; fmo++)
      Wd[fmo].array() += regularization.l2Penalty * W[fmo].array() / W[fmo].array().abs();
  }
  if(regularization.l2Penalty > 0.0)
  {
    for(int fmo = 0; fmo < fm; fmo++)
        Wd[fmo] += regularization.l2Penalty * W[fmo];
  }

  eout = &e;
}

Eigen::MatrixXd& Subsampling::getOutput()
{
  return y;
}

Eigen::VectorXd Subsampling::getParameters()
{
  Eigen::VectorXd p((1+bias)*fm*outRows*outCols);
  int idx = 0;
  for(int fmo = 0; fmo < fm; fmo++)
    for(int r = 0; r < outRows; r++)
      for(int c = 0; c < outCols; c++)
        p(idx++) = W[fmo](r, c);
  if(bias)
    for(int fmo = 0; fmo < fm; fmo++)
      for(int r = 0; r < outRows; r++)
        for(int c = 0; c < outCols; c++)
          p(idx++) = W[fmo](r, c);
  return p;
}

}
