#include <IntrinsicPlasticity.h>
#include <Random.h>
#include <io/DirectStorageDataSet.h>
#include <EigenWrapper.h>

namespace OpenANN
{

IntrinsicPlasticity::IntrinsicPlasticity(int nodes, fpt mu, fpt stdDev)
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
    b(i) = rng.sampleNormalDistribution<fpt>() * stdDev;
}

fpt IntrinsicPlasticity::error()
{
  fpt e = 0.0;
  const fpt N = dataSet->samples();
  for(int n = 0; n < N; n++)
    e += error(n);
  return e;
}

fpt IntrinsicPlasticity::error(unsigned int n)
{
  fpt e = 0.0;
  (*this)(dataSet->getInstance(n));
  for(int i = 0; i < nodes; i++)
  {
    const fpt ei = y(i) - mu;
    e += ei*ei;
  }
  return e;
}

Vt IntrinsicPlasticity::currentParameters()
{
  int i = 0;
  for(; i < nodes; i++)
    parameters(i) = s(i);
  for(int j = 0; j < nodes; j++, i++)
    parameters(i) = b(j);
  return parameters;
}

void IntrinsicPlasticity::setParameters(const Vt& parameters)
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

Vt IntrinsicPlasticity::gradient()
{
  g.fill(0.0);
  for(int n = 0; n < dataSet->samples(); n++)
    g += gradient(n);
  return g;
}

Vt IntrinsicPlasticity::gradient(unsigned int n)
{
  Vt a = dataSet->getInstance(n);
  OPENANN_CHECK_MATRIX_BROKEN(a);
  Vt y = (*this)(a);
  OPENANN_CHECK_MATRIX_BROKEN(y);

  Vt g(2*nodes);
  int i = 0;
  const fpt tmp = 2.0 + 1.0/mu;
  OPENANN_CHECK_NOT_EQUALS(s(i), (fpt) 0.0);
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

Mt IntrinsicPlasticity::hessian()
{
  return Mt::Identity(2*nodes, 2*nodes);
}

Learner& IntrinsicPlasticity::trainingSet(Mt& trainingInput, Mt& trainingOutput)
{
  if(deleteDataSet)
    delete dataSet;
  dataSet = new DirectStorageDataSet(trainingInput, trainingOutput);
  deleteDataSet = true;
}

Learner& IntrinsicPlasticity::trainingSet(DataSet& trainingSet)
{
  if(deleteDataSet)
    delete dataSet;
  dataSet = &trainingSet;
  deleteDataSet = false;
}

Vt IntrinsicPlasticity::operator()(const Vt& a)
{
  for(int i = 0; i < nodes; i++)
  {
    const fpt input = s(i) * a(i) + b(i);
    if(input > 45.0)
      y(i) = 1.0;
    else if(input < -45.0)
      y(i) = 0.0;
    else
      y(i) = 1.0 / (1.0 + exp(-input));
  }
  return y;
}

}
