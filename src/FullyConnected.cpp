#include <OpenANN/layers/FullyConnected.h>
#include <OpenANN/util/Random.h>

namespace OpenANN {

FullyConnected::FullyConnected(OutputInfo info, int J, bool bias,
                               ActivationFunction act, double stdDev,
                               double maxSquaredWeightNorm)
  : I(info.outputs()), J(J), bias(bias), act(act), stdDev(stdDev),
    maxSquaredWeightNorm(maxSquaredWeightNorm), W(J, I+bias), Wd(J, I+bias), x(0), a(J),
    y(J), yd(J), deltas(J), e(I)
{
}

OutputInfo FullyConnected::initialize(std::vector<double*>& parameterPointers,
                                      std::vector<double*>& parameterDerivativePointers)
{
  parameterPointers.reserve(parameterPointers.size() + J*(I+bias));
  parameterDerivativePointers.reserve(parameterDerivativePointers.size() + J*(I+bias));
  for(int j = 0; j < J; j++)
  {
    for(int i = 0; i < I+bias; i++)
    {
      parameterPointers.push_back(&W(j, i));
      parameterDerivativePointers.push_back(&Wd(j, i));
    }
  }

  initializeParameters();

  OutputInfo info;
  info.dimensions.push_back(J);
  return info;
}

void FullyConnected::initializeParameters()
{
  RandomNumberGenerator rng;
  for(int j = 0; j < J; j++)
    for(int i = 0; i < I+bias; i++)
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
  a = W.leftCols(I) * *x;
  if(bias)
    a += W.rightCols(1);
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
  Wd.leftCols(I) = deltas * x->transpose();
  if(bias)
    Wd.rightCols(1) = deltas;
  // Prepare error signals for previous layer
  e = W.leftCols(I).transpose() * deltas;
  eout = &e;
}

Eigen::VectorXd& FullyConnected::getOutput()
{
  return y;
}

}
