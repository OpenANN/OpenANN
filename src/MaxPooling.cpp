#include <OpenANN/layers/MaxPooling.h>
#include <OpenANN/util/AssertionMacros.h>
#include <limits>
#include <algorithm>

namespace OpenANN {

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
  outRows = inRows/kernelRows;
  outCols = inCols/kernelCols;
  fmOutSize = outRows * outCols;
  info.dimensions.push_back(outRows);
  info.dimensions.push_back(outCols);
  fmInSize = inRows * inCols;
  maxRow = inRows-kernelRows+1;
  maxCol = inCols-kernelCols+1;

  y.resize(1, info.outputs());
  deltas.resize(1, info.outputs());

  return info;
}

void MaxPooling::initializeParameters()
{
}

void MaxPooling::forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                                  bool dropout)
{
  this->x = x;

  OPENANN_CHECK(x->cols() == fm * inRows * inRows);
  OPENANN_CHECK_EQUALS(this->y.cols(), fm * outRows * outCols);

  int outputIdx = 0;
  int inputIdx = 0;
  for(int fmo = 0; fmo < fm; fmo++)
  {
    for(int ri = 0, ro = 0; ri < maxRow; ri+=kernelRows, ro++)
    {
      int rowBase = fmo*fmInSize + ri*inCols;
      for(int ci = 0, co = 0; ci < maxCol; ci+=kernelCols, co++, outputIdx++)
      {
        double m = -std::numeric_limits<double>::max();
        for(int kr = 0; kr < kernelRows; kr++)
        {
          inputIdx = rowBase + ci;
          for(int kc = 0; kc < kernelCols; kc++, inputIdx++)
            m = std::max(m, (*x)(0, inputIdx));
        }
        this->y(0, outputIdx) = m;
      }
    }
  }

  y = &(this->y);
}

void MaxPooling::backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout)
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
        double m = -std::numeric_limits<double>::max();
        int idx = -1;
        for(int kr = 0; kr < kernelRows; kr++)
        {
          inputIdx = rowBase + ci;
          for(int kc = 0; kc < kernelCols; kc++, inputIdx++)
            if((*x)(0, inputIdx) > m)
            {
              m = (*x)(0, inputIdx);
              idx = inputIdx;
            }
        }
        e(0, idx) = deltas(0, outputIdx);
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
