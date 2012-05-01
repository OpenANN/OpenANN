#include "ActivationFunctionsTestCase.h"
#include <ActivationFunctions.h>

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
  Vt a = Vt::Random(N);
  OpenANN::softmax(a);
  ASSERT_EQUALS_DELTA((fpt) 1.0, a.sum(), (fpt) 1e-3);
  ASSERT_WITHIN(a.minCoeff(), (fpt) 0.0, (fpt) 1.0);
  ASSERT_WITHIN(a.maxCoeff(), (fpt) 0.0, (fpt) 1.0);
}

void ActivationFunctionsTestCase::logistic()
{
  const int N = 1000;
  Vt a = Vt::Random(N) * (fpt) 10;
  Vt z = Vt::Zero(N);
  OpenANN::logistic(a, z);
  ASSERT_WITHIN(z.minCoeff(), (fpt) 0.0, (fpt) 0.2);
  ASSERT_WITHIN(z.maxCoeff(), (fpt) 0.8, (fpt) 1.0);

  Vt gd = Vt::Zero(N);
  OpenANN::logisticDerivative(z, gd);
  ASSERT_WITHIN(gd.minCoeff(), (fpt) 0.0, (fpt) 1.0);
  ASSERT_WITHIN(gd.maxCoeff(), (fpt) 0.0, (fpt) 1.0);
}

void ActivationFunctionsTestCase::normaltanh()
{
  const int N = 1000;
  Vt a = Vt::Random(N) * (fpt) 10;
  Vt z = Vt::Zero(N);
  OpenANN::normaltanh(a, z);
  ASSERT_WITHIN(z.minCoeff(), (fpt) -1.0, (fpt) -0.5);
  ASSERT_WITHIN(z.maxCoeff(), (fpt) 0.5, (fpt) 1.0);

  Vt gd = Vt::Zero(N);
  OpenANN::normaltanhDerivative(z, gd);
  ASSERT_WITHIN(gd.minCoeff(), (fpt) 0.0, (fpt) 1.0);
  ASSERT_WITHIN(gd.maxCoeff(), (fpt) 0.0, (fpt) 1.0);
}

void ActivationFunctionsTestCase::linear()
{
  const int N = 1000;
  Vt a = Vt::Random(N) * (fpt) 10;
  Vt z = Vt::Zero(N);
  OpenANN::linear(a, z);
  ASSERT_EQUALS(a.minCoeff(), z.minCoeff());
  ASSERT_EQUALS(a.maxCoeff(), z.maxCoeff());

  Vt gd = Vt::Zero(N);
  Vt expected = Vt::Ones(N);
  OpenANN::linearDerivative(gd);
  ASSERT_EQUALS(gd.sum(), expected.sum());
}
