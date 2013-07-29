#include "ZCATestCase.h"
#include <OpenANN/ZCAWhitening.h>
#include <OpenANN/util/Random.h>

void ZCATestCase::run()
{
  RUN(ZCATestCase, whiten);
}

void ZCATestCase::whiten()
{
  int N = 100;
  int D = 10;
  Eigen::MatrixXd X(N, D);
  OpenANN::RandomNumberGenerator rng;
  for(int n = 0; n < N; n++)
    for(int d = 0; d < D; d++)
      X(n, d) = rng.sampleNormalDistribution<double>();

  OpenANN::ZCAWhitening zca;
  Eigen::MatrixXd Y = zca.fit(X).transform(X);
  Eigen::MatrixXd C = 1.0 / (N-1.0) * Y.transpose() * Y;
  for(int d = 0; d < D; d++)
    ASSERT_EQUALS_DELTA(C(d, d), 1.0, 1e-5);
}
