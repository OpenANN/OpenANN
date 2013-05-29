#include <OpenANN/layers/MaxPooling.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/OpenANNException.h>
#include <limits>
#include <algorithm>

namespace OpenANN
{

MaxPooling::MaxPooling(OutputInfo info, int kernelRows, int kernelCols)
  : I(info.outputs()), fm(info.dimensions[0]),
    inRows(info.dimensions[1]), inCols(info.dimensions[2]),
    kernelRows(kernelRows), kernelCols(kernelCols), x(0), e(1, I)
{
}

OutputInfo MaxPooling::initialize(std::vector<double*>& parameterPointers,
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

  y.resize(1, info.outputs());
  deltas.resize(1, info.outputs());

  if(info.outputs() < 1)
    throw OpenANNException("Number of outputs in max-pooling layer is below"
                           " 1. You should either choose a smaller filter"
                           " size or generate a bigger input.");
  return info;
}

void MaxPooling::initializeParameters()
{
}

void MaxPooling::forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                                  bool dropout)
{
  const int N = x->rows();
  this->y.conservativeResize(N, Eigen::NoChange);
  this->x = x;

  OPENANN_CHECK(x->cols() == fm * inRows * inRows);
  OPENANN_CHECK_EQUALS(this->y.cols(), fm * outRows * outCols);

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
          double m = -std::numeric_limits<double>::max();
          for(int kr = 0; kr < kernelRows; kr++)
          {
            inputIdx = rowBase + ci;
            for(int kc = 0; kc < kernelCols; kc++, inputIdx++)
              m = std::max(m, (*x)(n, inputIdx));
          }
          this->y(n, outputIdx) = m;
        }
      }
    }
  }

  y = &(this->y);
}

void MaxPooling::backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout)
{
  const int N = y.rows();
  e.conservativeResize(N, Eigen::NoChange);
  deltas = (*ein);

  e.fill(0.0);
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
          double m = -std::numeric_limits<double>::max();
          int idx = -1;
          for(int kr = 0; kr < kernelRows; kr++)
          {
            inputIdx = rowBase + ci;
            for(int kc = 0; kc < kernelCols; kc++, inputIdx++)
              if((*x)(n, inputIdx) > m)
              {
                m = (*x)(n, inputIdx);
                idx = inputIdx;
              }
          }
          e(n, idx) = deltas(n, outputIdx);
        }
      }
    }
  }

  eout = &e;
}

Eigen::MatrixXd& MaxPooling::getOutput()
{
  return y;
}

}
