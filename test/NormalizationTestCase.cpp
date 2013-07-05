#include "NormalizationTestCase.h"
#include <OpenANN/Normalization.h>

void NormalizationTestCase::run()
{
  RUN(NormalizationTestCase, normalize);
}

void NormalizationTestCase::normalize()
{
  const int N = 100;
  const int D = 5;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  for(int n = 0; n < N; ++n)
    X(n, 0) = 0.0;
  OpenANN::Normalization n;
  n.fit(X);
  Eigen::MatrixXd Y = n.transform(X);
  n.fit(Y);
  Eigen::VectorXd m = n.getMean();
  Eigen::VectorXd s = n.getStd();
  for(int d = 0; d < D; d++)
    ASSERT_EQUALS_DELTA(m(d), 0.0, 1e-5);
  for(int d = 0; d < D; d++)
    ASSERT_EQUALS_DELTA(s(d), 1.0, 1e-5);
}
