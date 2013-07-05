#include "PCATestCase.h"
#include <OpenANN/Normalization.h>
#include <OpenANN/PCA.h>
#include <OpenANN/util/Random.h>

void PCATestCase::run()
{
  RUN(PCATestCase, pca);
}

void PCATestCase::pca()
{
  OpenANN::RandomNumberGenerator rng;
  const int N = 100;
  const int D = 2;
  Eigen::MatrixXd X(N, D);
  for(int n = 0; n < N; ++n)
  {
    for(int d = 0; d < D; ++d)
    {
      X(n, d) = rng.sampleNormalDistribution<double>();
    }
  }
  // Linear transformation
  Eigen::MatrixXd A(D, D);
  A << 1.0, 0.5, 0.5, 2.0;
  Eigen::MatrixXd Xt = X * A.transpose();
  // Decorrelation (without dimensionality reduction)
  OpenANN::PCA pca;
  pca.fit(Xt);
  Eigen::MatrixXd Y = pca.transform(Xt);

  // Covariance matrix should be identity matrix
  Eigen::MatrixXd cov = Y.transpose() * Y;
  ASSERT_EQUALS_DELTA(cov(0, 0), (double) N, 1e-5);
  ASSERT_EQUALS_DELTA(cov(1, 1), (double) N, 1e-5);
  ASSERT_EQUALS_DELTA(cov(0, 1), 0.0, 1e-5);
  ASSERT_EQUALS_DELTA(cov(1, 0), 0.0, 1e-5);
}
