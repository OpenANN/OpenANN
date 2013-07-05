#include "PCATestCase.h"
#include <OpenANN/Normalization.h>
#include <OpenANN/PCA.h>
#include <OpenANN/util/Random.h>

void PCATestCase::run()
{
  RUN(PCATestCase, decorrelation);
  RUN(PCATestCase, dimensionalityReduction);
}

void PCATestCase::decorrelation()
{
  OpenANN::RandomNumberGenerator rng;
  const int N = 100;
  const int D = 2;
  Eigen::MatrixXd X(N, D);
  for(int n = 0; n < N; ++n)
    for(int d = 0; d < D; ++d)
      X(n, d) = rng.sampleNormalDistribution<double>();

  // Linear transformation (correlation)
  Eigen::MatrixXd A(D, D);
  A << 1.0, 0.5, 0.5, 2.0;
  Eigen::MatrixXd Xt = X * A.transpose();
  // Decorrelation (without dimensionality reduction)
  OpenANN::PCA pca(D);
  pca.fit(Xt);
  Eigen::MatrixXd Y = pca.transform(Xt);

  // Covariance matrix should be identity matrix
  Eigen::MatrixXd cov = Y.transpose() * Y;
  ASSERT_EQUALS_DELTA(cov(0, 0), (double) N, 1e-5);
  ASSERT_EQUALS_DELTA(cov(1, 1), (double) N, 1e-5);
  ASSERT_EQUALS_DELTA(cov(0, 1), 0.0, 1e-5);
  ASSERT_EQUALS_DELTA(cov(1, 0), 0.0, 1e-5);

  Eigen::VectorXd evr = pca.explainedVarianceRatio();
  double evrSum = evr.sum();
  ASSERT_EQUALS_DELTA(evrSum, 1.0, 1e-5);
}

void PCATestCase::dimensionalityReduction()
{
  OpenANN::RandomNumberGenerator rng;
  const int N = 100;
  const int D = 5;
  Eigen::MatrixXd X(N, D);
  for(int n = 0; n < N; ++n)
    for(int d = 0; d < D; ++d)
      X(n, d) = rng.sampleNormalDistribution<double>();

  // Strong correlation
  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(D, D) * 0.5 +
      Eigen::MatrixXd::Ones(D, D);
  Eigen::MatrixXd Xt = X * A.transpose();
  // Dimensionality reduction
  OpenANN::PCA pca(1);
  pca.fit(Xt);
  Eigen::MatrixXd Y = pca.transform(Xt);
  ASSERT_EQUALS(Y.rows(), N);
  ASSERT_EQUALS(Y.cols(), 1);
  ASSERT_EQUALS(pca.explainedVarianceRatio().rows(), 1);
  ASSERT(pca.explainedVarianceRatio().sum() > 0.9);
}
