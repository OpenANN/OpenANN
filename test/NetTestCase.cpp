#include "NetTestCase.h"
#include "FiniteDifferences.h"
#include <OpenANN/Net.h>
#include <OpenANN/util/Random.h>

void NetTestCase::run()
{
  RUN(NetTestCase, dimension);
  RUN(NetTestCase, error);
  RUN(NetTestCase, gradientSSE);
  RUN(NetTestCase, gradientCE);
  RUN(NetTestCase, predictMinibatch);
}

void NetTestCase::dimension()
{
  const int D = 5;
  const int F = 2;
  const int N = 1;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  Eigen::MatrixXd T = Eigen::MatrixXd::Random(N, F);

  OpenANN::Net net;
  net.inputLayer(D)
     .fullyConnectedLayer(2, OpenANN::TANH)
     .outputLayer(F, OpenANN::LINEAR)
     .trainingSet(X, T);

  const int expectedDimension = 18;
  ASSERT_EQUALS(net.dimension(), expectedDimension);
  ASSERT_EQUALS(net.gradient().rows(), expectedDimension);
  double error;
  Eigen::VectorXd grad(net.dimension());
  net.errorGradient(0, error, grad);
  ASSERT_EQUALS(grad.rows(), expectedDimension);
  net.errorGradient(error, grad);
  ASSERT_EQUALS(grad.rows(), expectedDimension);
  ASSERT_EQUALS(net.currentParameters().rows(), expectedDimension);
}

void NetTestCase::error()
{
  const int D = 5;
  const int F = 2;
  const int N = 2;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  Eigen::MatrixXd T = Eigen::MatrixXd::Random(N, F);

  OpenANN::Net net;
  net.inputLayer(D)
     .fullyConnectedLayer(2, OpenANN::TANH)
     .outputLayer(F, OpenANN::LINEAR)
     .trainingSet(X, T);

  double error0 = net.error(0);
  double error1 = net.error(1);
  double error = net.error();
  ASSERT_EQUALS(error, error0 + error1);
}

void NetTestCase::gradientSSE()
{
  const int D = 5;
  const int F = 2;
  const int N = 2;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  Eigen::MatrixXd T = Eigen::MatrixXd::Random(N, F);

  OpenANN::Net net;
  net.inputLayer(D)
     .fullyConnectedLayer(2, OpenANN::TANH)
     .outputLayer(F, OpenANN::LINEAR)
     .trainingSet(X, T);

  Eigen::VectorXd ga0 = OpenANN::FiniteDifferences::parameterGradient(0, net);
  Eigen::VectorXd ga1 = OpenANN::FiniteDifferences::parameterGradient(1, net);
  Eigen::VectorXd ga = ga0 + ga1;

  Eigen::VectorXd g = net.gradient();
  double error;
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga(k), g(k), 1e-2);

  net.errorGradient(error, g);
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga(k), g(k), 1e-2);

  Eigen::VectorXd g0 = net.gradient(0);
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga0(k), g0(k), 1e-2);

  net.errorGradient(0, error, g0);
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga0(k), g0(k), 1e-2);
}

void NetTestCase::gradientCE()
{
  const int D = 5;
  const int F = 2;
  const int N = 2;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  // Target components have to sum up to 1
  Eigen::MatrixXd T(N, F);
  T.fill(0.0);
  OpenANN::RandomNumberGenerator rng;
  for(int n = 0; n < N; n++)
    T(n, rng.generateIndex(F)) = 1.0;

  OpenANN::Net net;
  net.inputLayer(D)
     .fullyConnectedLayer(2, OpenANN::TANH)
     .outputLayer(F, OpenANN::LINEAR)
     .setErrorFunction(OpenANN::CE)
     .trainingSet(X, T);

  Eigen::VectorXd ga0 = OpenANN::FiniteDifferences::parameterGradient(0, net);
  Eigen::VectorXd ga1 = OpenANN::FiniteDifferences::parameterGradient(1, net);
  Eigen::VectorXd ga = ga0 + ga1;

  Eigen::VectorXd g = net.gradient();
  double error;
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga(k), g(k), 1e-2);

  net.errorGradient(error, g);
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga(k), g(k), 1e-2);

  Eigen::VectorXd g0 = net.gradient(0);
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga0(k), g0(k), 1e-2);

  net.errorGradient(0, error, g0);
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga0(k), g0(k), 1e-2);
}

void NetTestCase::predictMinibatch()
{
  const int D = 5;
  const int F = 2;
  const int N = 2;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  Eigen::MatrixXd T = Eigen::MatrixXd::Random(N, F);

  OpenANN::Net net;
  net.inputLayer(D)
     .fullyConnectedLayer(2, OpenANN::TANH)
     .outputLayer(F, OpenANN::LINEAR)
     .trainingSet(X, T);

  Eigen::MatrixXd Y1(N, F);
  Eigen::VectorXd x(D);
  for(int n = 0; n < N; n++)
  {
    x = X.row(n);
    Y1.row(n) = net(x);
  }
  Eigen::MatrixXd Y2 = net(X);
  for(int n = 0; n < N; n++)
  {
    for(int f = 0; f < F; f++)
      ASSERT_EQUALS(Y1(n, f), Y2(n, f));
  }
}
