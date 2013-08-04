#include "ZCATestCase.h"
#include <OpenANN/ZCAWhitening.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/io/Logger.h>

void ZCATestCase::run()
{
  RUN(ZCATestCase, whiten);
}

void ZCATestCase::whiten()
{
  int N = 1000;
  int N2 = 500;
  int D = 5;
  Eigen::VectorXd w = Eigen::VectorXd::Random(D);
  Eigen::MatrixXd X(N, D);
  OpenANN::RandomNumberGenerator rng;
  rng.fillNormalDistribution(X);
  X *= w.asDiagonal();
  Eigen::MatrixXd X2(N2, D);
  rng.fillNormalDistribution(X2);
  X2 *= w.asDiagonal();

  OpenANN::ZCAWhitening zca;
  Eigen::MatrixXd Y = zca.fit(X).transform(X);
  Eigen::MatrixXd C = 1.0 / (N-1.0) * Y.transpose() * Y;
  for(int d = 0; d < D; d++)
    ASSERT_EQUALS_DELTA(C(d, d), 1.0, 1e-2);
  Eigen::MatrixXd Y2 = zca.transform(X2);
  C = 1.0 / (N2-1.0) * Y2.transpose() * Y2;
  for(int d = 0; d < D; d++)
    ASSERT_EQUALS_DELTA(C(d, d), 1.0, 0.15);
}
