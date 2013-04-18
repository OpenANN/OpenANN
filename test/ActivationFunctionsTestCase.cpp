#include "ActivationFunctionsTestCase.h"
#include <OpenANN/ActivationFunctions.h>

void ActivationFunctionsTestCase::run()
{
  RUN(ActivationFunctionsTestCase, softmax);
  RUN(ActivationFunctionsTestCase, logistic);
  RUN(ActivationFunctionsTestCase, normaltanh);
  RUN(ActivationFunctionsTestCase, linear);
}

void ActivationFunctionsTestCase::softmax()
{
  const int N = 1000;
  Eigen::VectorXd a = Eigen::VectorXd::Random(N);
  OpenANN::softmax(a);
  ASSERT_EQUALS_DELTA((double) 1.0, a.sum(), (double) 1e-3);
  ASSERT_WITHIN(a.minCoeff(), (double) 0.0, (double) 1.0);
  ASSERT_WITHIN(a.maxCoeff(), (double) 0.0, (double) 1.0);
}

void ActivationFunctionsTestCase::logistic()
{
  const int N = 1000;
  Eigen::VectorXd a = Eigen::VectorXd::Random(N) * (double) 10;
  Eigen::VectorXd z = Eigen::VectorXd::Zero(N);
  OpenANN::logistic(a, z);
  ASSERT_WITHIN(z.minCoeff(), (double) 0.0, (double) 0.2);
  ASSERT_WITHIN(z.maxCoeff(), (double) 0.8, (double) 1.0);

  Eigen::VectorXd gd = Eigen::VectorXd::Zero(N);
  OpenANN::logisticDerivative(z, gd);
  ASSERT_WITHIN(gd.minCoeff(), (double) 0.0, (double) 1.0);
  ASSERT_WITHIN(gd.maxCoeff(), (double) 0.0, (double) 1.0);
}

void ActivationFunctionsTestCase::normaltanh()
{
  const int N = 1000;
  Eigen::VectorXd a = Eigen::VectorXd::Random(N) * (double) 10;
  Eigen::VectorXd z = Eigen::VectorXd::Zero(N);
  OpenANN::normaltanh(a, z);
  ASSERT_WITHIN(z.minCoeff(), (double) -1.0, (double) -0.5);
  ASSERT_WITHIN(z.maxCoeff(), (double) 0.5, (double) 1.0);

  Eigen::VectorXd gd = Eigen::VectorXd::Zero(N);
  OpenANN::normaltanhDerivative(z, gd);
  ASSERT_WITHIN(gd.minCoeff(), (double) 0.0, (double) 1.0);
  ASSERT_WITHIN(gd.maxCoeff(), (double) 0.0, (double) 1.0);
}

void ActivationFunctionsTestCase::linear()
{
  const int N = 1000;
  Eigen::VectorXd a = Eigen::VectorXd::Random(N) * (double) 10;
  Eigen::VectorXd z = Eigen::VectorXd::Zero(N);
  OpenANN::linear(a, z);
  ASSERT_EQUALS(a.minCoeff(), z.minCoeff());
  ASSERT_EQUALS(a.maxCoeff(), z.maxCoeff());

  Eigen::VectorXd gd = Eigen::VectorXd::Zero(N);
  Eigen::VectorXd expected = Eigen::VectorXd::Ones(N);
  OpenANN::linearDerivative(gd);
  ASSERT_EQUALS(gd.sum(), expected.sum());
}
