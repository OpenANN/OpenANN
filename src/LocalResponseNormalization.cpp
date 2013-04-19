#include <OpenANN/layers/LocalResponseNormalization.h>
#include <OpenANN/util/Random.h>

namespace OpenANN {

LocalResponseNormalization::LocalResponseNormalization(
        OutputInfo info, bool bias, double k, int n, double alpha, double beta)
  : I(info.outputs()-bias), fm(info.dimensions[0]), rows(info.dimensions[1]),
    cols(info.dimensions[2]), bias(bias), x(0), denoms(I), y(I+bias), etmp(I),
    e(I), k(k), n(n), alpha(alpha), beta(beta)
{
}

OutputInfo LocalResponseNormalization::initialize(
    std::vector<double*>& parameterPointers,
    std::vector<double*>& parameterDerivativePointers)
{
  // Bias component will not change after initialization
  if(bias)
    y(I) = double(1.0);

  fmSize = rows*cols;

  OutputInfo info;
  info.bias = bias;
  info.dimensions.push_back(fm);
  info.dimensions.push_back(rows);
  info.dimensions.push_back(cols);
  return info;
}

void LocalResponseNormalization::forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout)
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
          register double a = (*x)(fmIn*fmSize+r*cols+c);
          denom += a*a;
        }
        denom = k + alpha*denom;
        denoms(outputIdx) = denom;
        this->y(outputIdx) = (*x)(outputIdx) * std::pow(denom, -beta);
      }
    }
  }
  y = &this->y;
}

void LocalResponseNormalization::backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout)
{
  for(int i = 0; i < I; i++)
    etmp(i) = (*ein)(i) * y(i) / denoms(i);
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
        e(outputIdx) = (*x)(outputIdx) * nom + (*ein)(outputIdx) *
            std::pow(denoms(outputIdx), -beta);
      }
    }
  }

  eout = &e;
}

Eigen::VectorXd& LocalResponseNormalization::getOutput()
{
  return y;
}

}
