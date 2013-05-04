#include <OpenANN/layers/Dropout.h>
#include <OpenANN/util/Random.h>

namespace OpenANN {

Dropout::Dropout(OutputInfo info, double dropoutProbability)
  : info(info), I(info.outputs()),
    dropoutProbability(dropoutProbability), y(I), dropoutMask(I), e(I)
{
}

OutputInfo Dropout::initialize(std::vector<double*>& parameterPointers,
                               std::vector<double*>& parameterDerivativePointers)
{
  return info;
}

void Dropout::forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout)
{
  dropoutMask.fill(1.0);
  if(dropout)
  {
    // Sample dropout mask
    RandomNumberGenerator rng;
    for(int i = 0; i < I; i++)
      if(rng.generate<double>(0.0, 1.0) < dropoutProbability)
        dropoutMask(i) = 0.0;
  }
  else if(dropoutProbability > 0.0)
  {
    // Scale down after training
    dropoutMask *= 1.0 - dropoutProbability;
  }
  this->y = dropoutMask.cwiseProduct(*x);
  y = &this->y;
}

void Dropout::backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout)
{
  e = dropoutMask.cwiseProduct(*ein);
  eout = &e;
}

Eigen::VectorXd& Dropout::getOutput()
{
  return y;
}

}
