#include <OpenANN/layers/Extreme.h>
#include <OpenANN/util/Random.h>

namespace OpenANN {

Extreme::Extreme(OutputInfo info, int J, bool bias, ActivationFunction act,
                 double stdDev)
  : I(info.outputs()), J(J), bias(bias), act(act), stdDev(stdDev),
    W(J, I), x(0), a(J), y(J+bias), yd(J), deltas(J), e(I)
{
}

OutputInfo Extreme::initialize(std::vector<double*>& parameterPointers,
                                      std::vector<double*>& parameterDerivativePointers)
{
  // Bias component will not change after initialization
  if(bias)
    y(J) = double(1.0);

  initializeParameters();

  OutputInfo info;
  info.bias = bias;
  info.dimensions.push_back(J);
  return info;
}

void Extreme::initializeParameters()
{
  RandomNumberGenerator rng;
  for(int j = 0; j < J; j++)
    for(int i = 0; i < I; i++)
      W(j, i) = rng.sampleNormalDistribution<double>() * stdDev;
}

void Extreme::forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout)
{
  this->x = x;
  // Activate neurons
  a = W * *x;
  // Compute output
  activationFunction(act, a, this->y);
  y = &(this->y);
}

void Extreme::backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout)
{
  // Derive activations
  activationFunctionDerivative(act, y, yd);
  for(int j = 0; j < J; j++)
    deltas(j) = yd(j) * (*ein)(j);
  // Prepare error signals for previous layer
  e = W.transpose() * deltas;
  eout = &e;
}

Eigen::VectorXd& Extreme::getOutput()
{
  return y;
}

}
