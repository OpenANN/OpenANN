#include "NetTestCase.h"
#include "FiniteDifferences.h"
#include <OpenANN/Net.h>

using namespace OpenANN;

void NetTestCase::run()
{
  RUN(NetTestCase, dimension);
  RUN(NetTestCase, gradient);
}

void NetTestCase::dimension()
{
  const int D = 5;
  const int F = 2;
  const int N = 1;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  Eigen::MatrixXd T = Eigen::MatrixXd::Random(N, F);

  Net net;
  net.inputLayer(D)
     .fullyConnectedLayer(2, TANH)
     .outputLayer(F, LINEAR)
     .trainingSet(X, T);

  const int expectedDimension = 18;
  ASSERT_EQUALS(net.dimension(), expectedDimension);
  ASSERT_EQUALS(net.gradient().rows(), expectedDimension);
  double error;
  Eigen::VectorXd grad(net.dimension());
  net.errorGradient(0, error, grad);
  ASSERT_EQUALS(grad.rows(), expectedDimension);
  ((Optimizable &) net).errorGradient(error, grad);
  ASSERT_EQUALS(grad.rows(), expectedDimension);
  ASSERT_EQUALS(net.currentParameters().rows(), expectedDimension);
}

void NetTestCase::gradient()
{
  const int D = 5;
  const int F = 2;
  const int N = 2;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  Eigen::MatrixXd T = Eigen::MatrixXd::Random(N, F);

  Net net;
  net.inputLayer(D)
     .fullyConnectedLayer(2, TANH)
     .outputLayer(F, LINEAR)
     .trainingSet(X, T);

  Eigen::VectorXd ga0 = OpenANN::FiniteDifferences::parameterGradient(0, net);
  Eigen::VectorXd ga1 = OpenANN::FiniteDifferences::parameterGradient(1, net);
  Eigen::VectorXd ga = ga0 + ga1;
  Eigen::VectorXd g = net.gradient();
  double error;
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga(k), g(k), 1e-2);

  ((Optimizable&) net).errorGradient(error, g);
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga(k), g(k), 1e-2);

  Eigen::VectorXd g0 = net.gradient(0);
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga0(k), g0(k), 1e-2);

  net.errorGradient(0, error, g0);
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga0(k), g0(k), 1e-2);
}
