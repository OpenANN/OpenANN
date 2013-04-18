#include <OpenANN/layers/FullyConnected.h>
#include <OpenANN/util/Random.h>

namespace OpenANN {

FullyConnected::FullyConnected(OutputInfo info, int J, bool bias,
                               ActivationFunction act, double stdDev,
                               double dropoutProbability,
                               double maxSquaredWeightNorm)
  : I(info.outputs()), J(J), bias(bias), act(act), stdDev(stdDev),
    dropoutProbability(dropoutProbability),
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
    y(J) = double(1.0);

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
  if(dropout)
  {
    RandomNumberGenerator rng;
    for(int j = 0; j < J; j++)
      if(rng.generate<double>(0.0, 1.0) < dropoutProbability)
        this->y(j) = (double) 0;
  }
  else if(dropoutProbability > 0.0)
  {
    // Hinton et al., 2012: "At test time, we use the "mean network" [...] to
    // compensate for the fact that [all] of them are active."
    // Scaling the outputs is equivalent to scaling the outgoing weights.
    this->y *= (1.0 - dropoutProbability);
    if(bias)
      this->y(J) = 1.0;
  }
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
