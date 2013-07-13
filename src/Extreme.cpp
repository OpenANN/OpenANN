#include <OpenANN/layers/Extreme.h>
#include <OpenANN/util/Random.h>

namespace OpenANN
{

Extreme::Extreme(OutputInfo info, int J, bool bias, ActivationFunction act,
                 double stdDev)
  : I(info.outputs()), J(J), bias(bias), act(act), stdDev(stdDev),
    W(J, I + bias), x(0), a(1, J), y(1, J), yd(1, J), deltas(1, J), e(1, I)
{
}

OutputInfo Extreme::initialize(std::vector<double*>& parameterPointers,
                               std::vector<double*>& parameterDerivativePointers)
{
  initializeParameters();

  OutputInfo info;
  info.dimensions.push_back(J);
  return info;
}

void Extreme::initializeParameters()
{
  RandomNumberGenerator rng;
  for(int j = 0; j < J; j++)
    for(int i = 0; i < I + bias; i++)
      W(j, i) = rng.sampleNormalDistribution<double>() * stdDev;
}

void Extreme::forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y, bool dropout)
{
  this->x = x;
  // Activate neurons
  a = *x * W.leftCols(I).transpose();
  if(bias)
    a += W.rightCols(1).transpose();
  // Compute output
  this->y.conservativeResize(a.rows(), Eigen::NoChange);
  activationFunction(act, a, this->y);
  y = &(this->y);
}

void Extreme::backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,
                            bool backpropToPrevious)
{
  // Derive activations
  this->yd.conservativeResize(a.rows(), Eigen::NoChange);
  activationFunctionDerivative(act, y, yd);
  deltas = yd.cwiseProduct(*ein);
  // Prepare error signals for previous layer
  if(backpropToPrevious)
    e = deltas * W.leftCols(I);
  eout = &e;
}

Eigen::MatrixXd& Extreme::getOutput()
{
  return y;
}

Eigen::VectorXd Extreme::getParameters()
{
  return Eigen::VectorXd();
}

}
