#include <OpenANN/layers/Dropout.h>
#include <OpenANN/util/Random.h>

namespace OpenANN
{

Dropout::Dropout(OutputInfo info, double dropoutProbability)
  : info(info), I(info.outputs()),
    dropoutProbability(dropoutProbability), y(1, I), dropoutMask(1, I), e(1, I)
{
}

OutputInfo Dropout::initialize(std::vector<double*>& parameterPointers,
                               std::vector<double*>& parameterDerivativePointers)
{
  return info;
}

void Dropout::forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y, bool dropout)
{
  const int N = x->rows();
  dropoutMask.conservativeResize(N, Eigen::NoChange);
  dropoutMask.fill(1.0);
  if(dropout)
  {
    // Sample dropout mask
    RandomNumberGenerator rng;
    for(int n = 0; n < N; n++)
    {
      for(int i = 0; i < I; i++)
        if(rng.generate<double>(0.0, 1.0) < dropoutProbability)
          dropoutMask(n, i) = 0.0;
    }
  }
  else if(dropoutProbability > 0.0)
  {
    // Scale down after training
    dropoutMask *= 1.0 - dropoutProbability;
  }
  this->y = dropoutMask.cwiseProduct(*x);
  y = &this->y;
}

void Dropout::backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,
                            bool backpropToPrevious)
{
  e = dropoutMask.cwiseProduct(*ein);
  eout = &e;
}

Eigen::MatrixXd& Dropout::getOutput()
{
  return y;
}

Eigen::VectorXd Dropout::getParameters()
{
  return Eigen::VectorXd();
}

}
