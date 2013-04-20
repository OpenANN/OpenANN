#include <OpenANN/layers/FullyConnected.h>
#include <OpenANN/util/Random.h>

namespace OpenANN {

FullyConnected::FullyConnected(OutputInfo info, int J, bool bias,
                               ActivationFunction act, double stdDev,
                               double maxSquaredWeightNorm)
  : I(info.outputs()), J(J), bias(bias), act(act), stdDev(stdDev),
    maxSquaredWeightNorm(maxSquaredWeightNorm), W(J, I), Wd(J, I), x(0), a(J),
    y(J+bias), yd(J), deltas(J), e(I)
{
}

OutputInfo FullyConnected::initialize(std::vector<double*>& parameterPointers,
                                      std::vector<double*>& parameterDerivativePointers)
{
  parameterPointers.reserve(parameterPointers.size() + J*I);
  parameterDerivativePointers.reserve(parameterDerivativePointers.size() + J*I);
  for(int j = 0; j < J; j++)
  {
    for(int i = 0; i < I; i++)
    {
      parameterPointers.push_back(&W(j, i));
      parameterDerivativePointers.push_back(&Wd(j, i));
    }
  }

  // Bias component will not change after initialization
  if(bias)
    y(J) = 1.0;

  initializeParameters();

  OutputInfo info;
  info.bias = bias;
  info.dimensions.push_back(J);
  return info;
}

void FullyConnected::initializeParameters()
{
  RandomNumberGenerator rng;
  for(int j = 0; j < J; j++)
    for(int i = 0; i < I; i++)
      W(j, i) = rng.sampleNormalDistribution<double>() * stdDev;
}

void FullyConnected::updatedParameters()
{
  if(maxSquaredWeightNorm > 0.0)
  {
    for(int j = 0; j < J; j++)
    {
      const double squaredNorm = W.row(j).squaredNorm();
      if(squaredNorm > maxSquaredWeightNorm)
        W.row(j) *= sqrt(maxSquaredWeightNorm / squaredNorm);
    }
  }
}

void FullyConnected::forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout)
{
  this->x = x;
  // Activate neurons
  a = W * *x;
  // Compute output
  activationFunction(act, a, this->y);
  y = &(this->y);
}

void FullyConnected::backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout)
{
  // Derive activations
  activationFunctionDerivative(act, y, yd);
  for(int j = 0; j < J; j++)
    deltas(j) = yd(j) * (*ein)(j);
  // Weight derivatives
  Wd = deltas * x->transpose();
  // Prepare error signals for previous layer
  e = W.transpose() * deltas;
  eout = &e;
}

Eigen::VectorXd& FullyConnected::getOutput()
{
  return y;
}

}
