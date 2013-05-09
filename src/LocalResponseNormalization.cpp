#include <OpenANN/layers/LocalResponseNormalization.h>
#include <OpenANN/util/Random.h>

namespace OpenANN {

LocalResponseNormalization::LocalResponseNormalization(
        OutputInfo info, double k, int n, double alpha, double beta)
  : I(info.outputs()), fm(info.dimensions[0]), rows(info.dimensions[1]),
    cols(info.dimensions[2]), x(0), denoms(1, I), y(1, I), etmp(1, I),
    e(1, I), k(k), n(n), alpha(alpha), beta(beta)
{
}

OutputInfo LocalResponseNormalization::initialize(
    std::vector<double*>& parameterPointers,
    std::vector<double*>& parameterDerivativePointers)
{
  fmSize = rows*cols;

  OutputInfo info;
  info.dimensions.push_back(fm);
  info.dimensions.push_back(rows);
  info.dimensions.push_back(cols);
  return info;
}

void LocalResponseNormalization::forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y, bool dropout)
{
  this->x = x;
  for(int fmOut=0, outputIdx=0; fmOut < fm; fmOut++)
  {
    for(int r = 0; r < rows; r++)
    {
      for(int c = 0; c < cols; c++, outputIdx++)
      {
        double denom = 0.0;
        const int fmInMin = std::max(0, fmOut-n/2);
        const int fmInMax = std::min(fm-1, fmOut+n/2);
        for(int fmIn=fmInMin; fmIn < fmInMax; fmIn++)
        {
          register double a = (*x)(0, fmIn*fmSize+r*cols+c);
          denom += a*a;
        }
        denom = k + alpha*denom;
        denoms(0, outputIdx) = denom;
        this->y(0, outputIdx) = (*x)(0, outputIdx) * std::pow(denom, -beta);
      }
    }
  }
  y = &this->y;
}

void LocalResponseNormalization::backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout)
{
  for(int i = 0; i < I; i++)
    etmp(0, i) = (*ein)(0, i) * y(0, i) / denoms(0, i);
  etmp *= -2.0*alpha*beta;

  for(int fmOut=0, outputIdx=0; fmOut < fm; fmOut++)
  {
    for(int r = 0; r < rows; r++)
    {
      for(int c = 0; c < cols; c++, outputIdx++)
      {
        double nom = 0.0;
        const int fmInMin = std::max(0, fmOut-n/2);
        const int fmInMax = std::min(fm-1, fmOut+n/2);
        for(int fmIn=fmInMin; fmIn < fmInMax; fmIn++)
          nom += etmp(fmIn*fmSize+r*cols+c);
        e(0, outputIdx) = (*x)(0, outputIdx) * nom + (*ein)(0, outputIdx) *
            std::pow(denoms(0, outputIdx), -beta);
      }
    }
  }

  eout = &e;
}

Eigen::MatrixXd& LocalResponseNormalization::getOutput()
{
  return y;
}

}
