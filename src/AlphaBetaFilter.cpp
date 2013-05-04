#include <OpenANN/layers/AlphaBetaFilter.h>
#include <OpenANN/util/Random.h>

namespace OpenANN {

AlphaBetaFilter::AlphaBetaFilter(OutputInfo info, double deltaT, double stdDev)
  : I(info.outputs()), J(2*I), deltaT(deltaT), stdDev(stdDev), gamma(I),
    gammad(I), alpha(I), beta(I), first(true), x(0), y(J)
{
}

OutputInfo AlphaBetaFilter::initialize(std::vector<double*>& parameterPointers,
                                       std::vector<double*>& parameterDerivativePointers)
{
  parameterPointers.reserve(parameterPointers.size() + I);
  parameterDerivativePointers.reserve(parameterDerivativePointers.size() + I);
  for(int i = 0; i < I; i++)
  {
    parameterPointers.push_back(&gamma(i));
    parameterDerivativePointers.push_back(&gammad(i));
  }

  initializeParameters();

  OutputInfo info;
  info.dimensions.push_back(J);
  return info;
}

void AlphaBetaFilter::initializeParameters()
{
  RandomNumberGenerator rng;
  for(int i = 0; i < I; i++)
  {
    gamma(i) = rng.sampleNormalDistribution<double>() * stdDev;
    gammad(i) = 0.0;
  }
}

void AlphaBetaFilter::updatedParameters()
{
  for(int i = 0; i < I; i++)
  {
    gamma(i) = fabs(gamma(i));
    const double r = (4.0 + gamma(i) - sqrt(8.0 * gamma(i) + gamma(i) * gamma(i))) / 4.0;
    alpha(i) = 1.0 - r*r;
    const double rr = 1.0 - r;
    beta(i) = 2.0 * rr * rr;
  }
  reset();
}

void AlphaBetaFilter::reset()
{
  first = true;
  y.fill(0.0);
}

void AlphaBetaFilter::forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout)
{
  this->x = x;

  if(first)
  {
    for(int i = 0, j = 0; i < I; i++, j+=2)
      this->y(j) = (*x)(i);
    first = false;
  }

  for(int i = 0, j = 0; i < I; i++, j+=2)
  {
    const double diff = (*x)(i) - this->y(j);
    this->y(j+1) += beta(i) / deltaT * diff;
    this->y(j) += alpha(i) * diff + deltaT * this->y(j+1);
  }

  y = &(this->y);
}

void AlphaBetaFilter::backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout)
{
  // Do nothing.
}

Eigen::VectorXd& AlphaBetaFilter::getOutput()
{
  return y;
}

}
