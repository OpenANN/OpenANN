#include <layers/FullyConnected.h>
#include <Random.h>

namespace OpenANN {

FullyConnected::FullyConnected(OutputInfo info, int J, bool bias, ActivationFunction act, fpt stdDev)
  : debugLogger(Logger::CONSOLE), I(info.outputs()), J(J), bias(bias),
    act(act), stdDev(stdDev), W(J, I), Wd(J, I), x(0), a(J), y(J+bias), yd(J),
    deltas(J), e(I)
{
}

OutputInfo FullyConnected::initialize(std::list<fpt*>& parameterPointers,
                                      std::list<fpt*>& parameterDerivativePointers)
{
  RandomNumberGenerator rng;
  for(int j = 0; j < J; j++)
    for(int i = 0; i < I; i++)
    {
      W(j, i) = rng.sampleNormalDistribution<fpt>() * stdDev;
      parameterPointers.push_back(&W(j, i));
      parameterDerivativePointers.push_back(&Wd(j, i));
    }
  // Bias component will not change after initialization
  if(bias)
    y(J) = fpt(1.0);
  OutputInfo info;
  info.bias = bias;
  info.dimensions.push_back(J);
  return info;
}

void FullyConnected::forwardPropagate(Vt* x, Vt*& y)
{
  this->x = x;
  // Activate neurons
  a = W * *x;
  // Compute output
  activationFunction(act, a, this->y);
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
