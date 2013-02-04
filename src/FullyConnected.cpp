#include <layers/FullyConnected.h>
#include <Random.h>

namespace OpenANN {

FullyConnected::FullyConnected(OutputInfo info, int J, bool bias,
                               ActivationFunction act, fpt stdDev,
                               fpt dropoutProbability)
  : debugLogger(Logger::CONSOLE), I(info.outputs()), J(J), bias(bias),
    act(act), stdDev(stdDev), dropoutProbability(dropoutProbability), W(J, I),
    Wd(J, I), x(0), a(J), y(J+bias), yd(J), deltas(J), e(I)
{
}

OutputInfo FullyConnected::initialize(std::vector<fpt*>& parameterPointers,
                                      std::vector<fpt*>& parameterDerivativePointers)
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
    y(J) = fpt(1.0);

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
      W(j, i) = rng.sampleNormalDistribution<fpt>() * stdDev;
}

void FullyConnected::forwardPropagate(Vt* x, Vt*& y, bool dropout)
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
      if(rng.generate<fpt>(0.0, 1.0) < dropoutProbability)
        this->y(j) = (fpt) 0;
  }
  y = &(this->y);
}

void FullyConnected::backpropagate(Vt* ein, Vt*& eout)
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

}
