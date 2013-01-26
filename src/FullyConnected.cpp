#include <layers/FullyConnected.h>
#include <Random.h>

namespace OpenANN {

FullyConnected::FullyConnected(int I, int J, bool bias, ActivationFunction act, fpt stdDev)
  : debugLogger(Logger::CONSOLE), I(I), J(J), bias(bias), act(act),
    stdDev(stdDev), W(J, I), Wd(J, I), x(0), a(J), y(J+bias), yd(J), deltas(J),
    e(I)
{
}

void FullyConnected::initialize(std::list<fpt*>& parameterPointers,
                                std::list<fpt*>& parameterDerivativePointers)
{
  RandomNumberGenerator rng;
  parameterPointers.clear();
  parameterDerivativePointers.clear();
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
}

void FullyConnected::forwardPropagate(Vt* x, Vt*& y)
{
  this->x = x;
  a = W * *x;
  activationFunction(act, a, this->y);
  y = &(this->y);
}

void FullyConnected::accumulate(Vt* e)
{
  activationFunctionDerivative(act, y, yd);
  for(int j = 0; j < J; j++)
    deltas(j) = yd(j) * (*e)(j);
}

void FullyConnected::gradient()
{
  Wd = deltas * x->transpose();
}

void FullyConnected::backpropagate(Vt*& e)
{
  this->e = W.transpose() * deltas;
  e = &(this->e);
}

}
