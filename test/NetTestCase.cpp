#include "NetTestCase.h"
#include "FiniteDifferences.h"
#include <OpenANN/Net.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/util/Random.h>
#include <sstream>

void NetTestCase::run()
{
  RUN(NetTestCase, dimension);
  RUN(NetTestCase, error);
  RUN(NetTestCase, gradientSSE);
  RUN(NetTestCase, gradientCE);
  RUN(NetTestCase, multilayerNetwork);
  RUN(NetTestCase, predictMinibatch);
  RUN(NetTestCase, minibatchErrorGradient);
  RUN(NetTestCase, saveLoad);
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

  double error0 = net.error(0) / (double) N;
  double error1 = net.error(1) / (double) N;
  double error = net.error();
  ASSERT_EQUALS(net.examples(), N);
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
  Eigen::VectorXd ga = (ga0 + ga1) / (double) N;

  Eigen::VectorXd g = net.gradient();
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga(k), g(k), 1e-2);

  double error;
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
  T.setZero();
  OpenANN::RandomNumberGenerator rng;
  for(int n = 0; n < N; n++)
    T(n, rng.generateIndex(F)) = 1.0;

  OpenANN::Net net;
  net.inputLayer(D)
  .fullyConnectedLayer(2, OpenANN::TANH)
  .outputLayer(F, OpenANN::SOFTMAX)
  .setErrorFunction(OpenANN::CE)
  .trainingSet(X, T);

  Eigen::VectorXd ga0 = OpenANN::FiniteDifferences::parameterGradient(0, net);
  Eigen::VectorXd ga1 = OpenANN::FiniteDifferences::parameterGradient(1, net);
  Eigen::VectorXd ga = (ga0 + ga1) / (double) N;

  Eigen::VectorXd g = net.gradient();
  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga(k), g(k), 1e-2);

  double error;
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

void NetTestCase::multilayerNetwork()
{
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(1, 1 * 6 * 6);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(1, 3);
  OpenANN::DirectStorageDataSet ds(&X, &Y);

  OpenANN::Net net;
  net.inputLayer(1, 6, 6);
  net.convolutionalLayer(4, 3, 3, OpenANN::TANH, 0.5);
  net.localReponseNormalizationLayer(2.0, 3, 0.01, 0.75);
  net.subsamplingLayer(2, 2, OpenANN::TANH, 0.5);
  net.fullyConnectedLayer(10, OpenANN::TANH, 0.5);
  net.extremeLayer(10, OpenANN::TANH, 0.05);
  net.outputLayer(3, OpenANN::LINEAR, 0.5);
  net.trainingSet(ds);

  Eigen::VectorXd g = net.gradient();
  Eigen::VectorXd e = OpenANN::FiniteDifferences::parameterGradient(0, net);
  double delta = std::max<double>(1e-2, 1e-5 * e.norm());
  for(int j = 0; j < net.dimension(); j++)
    ASSERT_EQUALS_DELTA(g(j), e(j), delta);
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

void NetTestCase::minibatchErrorGradient()
{
  const int D = 5;
  const int F = 2;
  const int N = 5;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  Eigen::MatrixXd T = Eigen::MatrixXd::Random(N, F);

  OpenANN::Net net;
  net.inputLayer(D)
  .fullyConnectedLayer(2, OpenANN::TANH)
  .outputLayer(F, OpenANN::LINEAR)
  .trainingSet(X, T);

  std::vector<int> indices;
  for(int n = 0; n < N; n++)
    indices.push_back(n);
  double error1, error2;

  Eigen::VectorXd g1(net.dimension()), g2(net.dimension());
  net.errorGradient(indices.begin(), indices.end(), error1, g1);
  net.errorGradient(error2, g2);

  for(int k = 0; k < net.dimension(); k++)
    ASSERT_EQUALS_DELTA(g1(k), g2(k), 1e-2);
  ASSERT_EQUALS_DELTA(error1, error2, 1e-2);
}

void NetTestCase::saveLoad()
{
  OpenANN::RandomNumberGenerator().seed(0);
  OpenANN::Net net;
  net.setRegularization(0.001, 0.001, 0.0);
  net.inputLayer(2, 6, 6)
  .convolutionalLayer(2, 3, 3, OpenANN::TANH)
  .subsamplingLayer(2, 2, OpenANN::TANH)
  .extremeLayer(5, OpenANN::TANH, 1.0)
  .fullyConnectedLayer(10, OpenANN::TANH)
  .sparseAutoEncoderLayer(5, 3.0, 0.1, OpenANN::LOGISTIC)
  .compressedOutputLayer(2, 2, OpenANN::SOFTMAX, "gaussian");
  net.setErrorFunction(OpenANN::CE);
  std::stringstream stream;
  net.save(stream);
  OpenANN::RandomNumberGenerator().seed(0);
  OpenANN::Net loadedNet;
  loadedNet.load(stream);
  ASSERT_EQUALS(net.numberOflayers(), loadedNet.numberOflayers());
  ASSERT_EQUALS(net.dimension(), loadedNet.dimension());
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 2*6*6);
  Eigen::MatrixXd Y1 = net(X);
  Eigen::MatrixXd Y2 = loadedNet(X);
  for(int n = 0; n < Y1.rows(); n++)
    for(int f = 0; f < Y2.cols(); f++)
      ASSERT_EQUALS_DELTA(Y1(n, f), Y2(n, f), 1e-5);
}
