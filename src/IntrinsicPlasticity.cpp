#include <OpenANN/IntrinsicPlasticity.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/util/EigenWrapper.h>

namespace OpenANN
{

IntrinsicPlasticity::IntrinsicPlasticity(int nodes, double mu, double stdDev)
    : nodes(nodes), mu(mu), stdDev(stdDev), s(nodes), b(nodes),
      parameters(2*nodes), g(2*nodes), dataSet(0), deleteDataSet(false),
      y(nodes)
{
}

IntrinsicPlasticity::~IntrinsicPlasticity()
{
  if(deleteDataSet)
    delete dataSet;
}

unsigned int IntrinsicPlasticity::examples()
{
  return dataSet->samples();
}

unsigned int IntrinsicPlasticity::dimension()
{
  return 2*nodes;
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
  const double N = dataSet->samples();
  for(int n = 0; n < N; n++)
    e += error(n);
  return e;
}

double IntrinsicPlasticity::error(unsigned int n)
{
  double e = 0.0;
  (*this)(dataSet->getInstance(n));
  for(int i = 0; i < nodes; i++)
  {
    const double ei = y(i) - mu;
    e += ei*ei;
  }
  return e;
}

Eigen::VectorXd IntrinsicPlasticity::currentParameters()
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
  g.fill(0.0);
  for(int n = 0; n < dataSet->samples(); n++)
    g += gradient(n);
  return g;
}

Eigen::VectorXd IntrinsicPlasticity::gradient(unsigned int n)
{
  Eigen::VectorXd a = dataSet->getInstance(n);
  OPENANN_CHECK_MATRIX_BROKEN(a);
  Eigen::VectorXd y = (*this)(a);
  OPENANN_CHECK_MATRIX_BROKEN(y);

  Eigen::VectorXd g(2*nodes);
  int i = 0;
  const double tmp = 2.0 + 1.0/mu;
  OPENANN_CHECK_NOT_EQUALS(s(i), (double) 0.0);
  for(; i < nodes; i++)
    g(i) = 1.0/s(i) + a(i) - tmp*a(i)*y(i) + a(i)*y(i)*y(i)/mu;
  for(int j = 0; j < nodes; j++, i++)
    g(i) = 1.0 - tmp*y(j) + y(j)*y(j)/mu;
  return -g; // Allows using gradient descent algorithms
}

bool IntrinsicPlasticity::providesHessian()
{
  return false;
}

Eigen::MatrixXd IntrinsicPlasticity::hessian()
{
  return Eigen::MatrixXd::Identity(2*nodes, 2*nodes);
}

Learner& IntrinsicPlasticity::trainingSet(Eigen::MatrixXd& trainingInput, Eigen::MatrixXd& trainingOutput)
{
  if(deleteDataSet)
    delete dataSet;
  dataSet = new DirectStorageDataSet(&trainingInput, &trainingOutput);
  deleteDataSet = true;
}

Learner& IntrinsicPlasticity::trainingSet(DataSet& trainingSet)
{
  if(deleteDataSet)
    delete dataSet;
  dataSet = &trainingSet;
  deleteDataSet = false;
}

Eigen::VectorXd IntrinsicPlasticity::operator()(const Eigen::VectorXd& a)
{
  for(int i = 0; i < nodes; i++)
  {
    const double input = s(i) * a(i) + b(i);
    if(input > 45.0)
      y(i) = 1.0;
    else if(input < -45.0)
      y(i) = 0.0;
    else
      y(i) = 1.0 / (1.0 + exp(-input));
  }
  return y;
}

Eigen::MatrixXd IntrinsicPlasticity::operator()(const Eigen::MatrixXd& A)
{
  // TODO vectorize implementation
  Eigen::MatrixXd Y(A.rows(), A.cols());
  for(int n = 0; n < A.rows(); n++)
  {
    for(int i = 0; i < nodes; i++)
    {
      const double input = s(i) * A(n, i) + b(i);
      if(input > 45.0)
        Y(n, i) = 1.0;
      else if(input < -45.0)
        Y(n, i) = 0.0;
      else
        Y(n, i) = 1.0 / (1.0 + exp(-input));
    }
  }
  return Y;
}

}
