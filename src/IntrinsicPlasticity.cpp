#include <OpenANN/IntrinsicPlasticity.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/util/EigenWrapper.h>
#include <OpenANN/ActivationFunctions.h>

namespace OpenANN
{

IntrinsicPlasticity::IntrinsicPlasticity(int nodes, double mu, double stdDev)
  : nodes(nodes), mu(mu), stdDev(stdDev), s(nodes), b(nodes),
    parameters(2 * nodes), g(2 * nodes), y(nodes)
{
  initialize();
}

unsigned int IntrinsicPlasticity::examples()
{
  return trainSet->samples();
}

unsigned int IntrinsicPlasticity::dimension()
{
  return 2 * nodes;
}

bool IntrinsicPlasticity::providesInitialization()
{
  return true;
}

void IntrinsicPlasticity::initialize()
{
  s.fill(1.0);
  RandomNumberGenerator rng;
  for(int i = 0; i < nodes; i++)
    b(i) = rng.sampleNormalDistribution<double>() * stdDev;
}

double IntrinsicPlasticity::error()
{
  double e = 0.0;
  const double N = trainSet->samples();
  for(int n = 0; n < N; n++)
    e += error(n);
  return e;
}

double IntrinsicPlasticity::error(unsigned int n)
{
  double e = 0.0;
  y = (*this)(trainSet->getInstance(n));
  for(int i = 0; i < nodes; i++)
  {
    const double ei = y(i) - mu;
    e += ei * ei;
  }
  return e;
}

const Eigen::VectorXd& IntrinsicPlasticity::currentParameters()
{
  int i = 0;
  for(; i < nodes; i++)
    parameters(i) = s(i);
  for(int j = 0; j < nodes; j++, i++)
    parameters(i) = b(j);
  return parameters;
}

void IntrinsicPlasticity::setParameters(const Eigen::VectorXd& parameters)
{
  int i = 0;
  for(; i < nodes; i++)
    s(i) = parameters(i);
  for(int j = 0; j < nodes; j++, i++)
    b(j) = parameters(i);
}

bool IntrinsicPlasticity::providesGradient()
{
  return true;
}

Eigen::VectorXd IntrinsicPlasticity::gradient()
{
  g.setZero();
  for(int n = 0; n < trainSet->samples(); n++)
    g += gradient(n);
  return g;
}

Eigen::VectorXd IntrinsicPlasticity::gradient(unsigned int n)
{
  Eigen::VectorXd a = trainSet->getInstance(n);
  OPENANN_CHECK_MATRIX_BROKEN(a);
  Eigen::VectorXd y = (*this)(a);
  OPENANN_CHECK_MATRIX_BROKEN(y);

  Eigen::VectorXd g(2 * nodes);
  int i = 0;
  const double tmp = 2.0 + 1.0 / mu;
  OPENANN_CHECK_NOT_EQUALS(s(i), (double) 0.0);
  for(; i < nodes; i++)
    g(i) = 1.0 / s(i) + a(i) - tmp * a(i) * y(i) + a(i) * y(i) * y(i) / mu;
  for(int j = 0; j < nodes; j++, i++)
    g(i) = 1.0 - tmp * y(j) + y(j) * y(j) / mu;
  return -g; // Allows using gradient descent algorithms
}

Eigen::VectorXd IntrinsicPlasticity::operator()(const Eigen::VectorXd& a)
{
  Eigen::MatrixXd A = a.transpose();
  return (*this)(A).transpose();
}

Eigen::MatrixXd IntrinsicPlasticity::operator()(const Eigen::MatrixXd& A)
{
  Y.conservativeResize(A.rows(), A.cols());
  for(int n = 0; n < A.rows(); n++)
    Y.row(n) = A.row(n).cwiseProduct(s.transpose()) + b.transpose();
  activationFunction(LOGISTIC, Y, Y);
  return Y;
}

OutputInfo IntrinsicPlasticity::initialize(std::vector<double*>& parameterPointers,
                                           std::vector<double*>& parameterDerivativePointers)
{
  OutputInfo info;
  info.dimensions.push_back(nodes);
  return info;
}

void IntrinsicPlasticity::forwardPropagate(Eigen::MatrixXd* x,
                                           Eigen::MatrixXd*& y, bool dropout)
{
  (*this)(*x);
  y = &Y;
}

void IntrinsicPlasticity::backpropagate(Eigen::MatrixXd* ein,
                                        Eigen::MatrixXd*& eout,
                                        bool backpropToPrevious)
{
  const int N = Y.rows();
  e.conservativeResize(N, nodes);
  Yd.conservativeResize(N, nodes);
  // Derive activations
  activationFunctionDerivative(LOGISTIC, Y, Yd);
  for(int n = 0; n < N; n++)
    Yd.row(n) = Yd.row(n).cwiseProduct(s.transpose());
  // Prepare error signals for previous layer
  if(backpropToPrevious)
    e = Yd.cwiseProduct(*ein);
  eout = &e;
}

Eigen::MatrixXd& IntrinsicPlasticity::getOutput()
{
  return Y;
}

Eigen::VectorXd IntrinsicPlasticity::getParameters()
{
  return currentParameters();
}

}
