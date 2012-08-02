#include <GP.h>
#include <io/DirectStorageDataSet.h>
#include <AssertionMacros.h>

namespace OpenANN {

GP::GP(fpt beta, fpt theta0, fpt theta1, fpt theta2, fpt theta3)
    : parameters(5), beta(parameters(0)), theta0(parameters(1)),
    theta1(parameters(1)), theta2(parameters(2)), theta3(parameters(3)),
    dataSet(0), deleteDataSetOnDestruction(false)
{
  this->beta = beta;
  this->theta0 = theta0;
  this->theta1 = theta1;
  this->theta2 = theta2;
  this->theta3 = theta3;
}

GP::~GP()
{
  if(deleteDataSetOnDestruction && dataSet)
    delete dataSet;
}

Vt GP::currentParameters()
{
  return parameters;
}

unsigned int GP::dimension()
{
  return parameters.size();
}

fpt GP::error()
{
  return 0.0; // TODO implement
}

Vt GP::gradient()
{
  OPENANN_CHECK(false && "GP: Gradient not available.");
  return Vt(1);
}

Mt GP::hessian()
{
  OPENANN_CHECK(false && "GP: Hessian not available.");
  return Mt(1, 1);
}

void GP::initialize()
{
  // TODO implement
}

bool GP::providesGradient()
{
  return false;
}

bool GP::providesHessian()
{
  return false;
}

bool GP::providesInitialization()
{
  return true;
}

void GP::setParameters(const Vt& parameters)
{
  this-> parameters = parameters;
}

Learner& GP::trainingSet(Mt& trainingInput, Mt& trainingOutput)
{
  deleteDataSetOnDestruction = true;
  dataSet = new DirectStorageDataSet(trainingInput, trainingOutput);
}

Learner& GP::trainingSet(DataSet& trainingSet)
{
  deleteDataSetOnDestruction = false;
  dataSet = &trainingSet;
}

void GP::buildModel()
{
  const int N = dataSet->samples();
  Mt covariance(N, N);
  t.resize(N, dataSet->inputs());
  for(int n = 0; n < N; n++)
  {
    for(int m = 0; m <= n; m++)
    {
      covariance(n, m) = kernel(dataSet->getInstance(n),
          dataSet->getInstance(m)) + (n == m) ? (fpt) 1.0 / beta : (fpt) 0.0;
      covariance(n, m) = covariance(m, n);
    }
    t.row(n) = dataSet->getTarget(n);
  }
  covarianceInv.resize(N, N);
  covarianceInv = covariance.inverse();
}

Vt GP::operator()(const Vt& x)
{
  const int N = dataSet->samples();
  Mt k(N, 1);
  for(int n = 0; n < N; n++)
    k(n, 0) = kernel(dataSet->getInstance(n), x);
  return k.transpose() * covarianceInv * t;
}

fpt GP::kernel(const Vt& x1, const Vt& x2)
{
  return theta0 * std::exp(-theta1/(fpt)2.0*(x1-x2).squaredNorm() + theta2 +
      theta3 * x1.dot(x2));
}

}
