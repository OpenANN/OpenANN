#include "KMeansTestCase.h"
#include <OpenANN/KMeans.h>
#include <OpenANN/util/Random.h>
#include <limits>

void KMeansTestCase::run()
{
  RUN(KMeansTestCase, clustering);
}

void KMeansTestCase::clustering()
{
  const int N = 1000;
  const int D = 10;
  Eigen::MatrixXd X(N, D);
  OpenANN::RandomNumberGenerator rng;
  rng.fillNormalDistribution(X);

  OpenANN::KMeans kmeans(D, 5);
  const int batchSize = 200;
  double averageDistToCenter = std::numeric_limits<double>::max();
  for(int i = 0; i < N/batchSize; i++)
  {
    // Data points will be closer to centers after each update
    Eigen::MatrixXd Y = kmeans.fitPartial(X.block(i*batchSize, 0, batchSize, D)).transform(X);
    const double newDistance = Y.array().rowwise().maxCoeff().sum();
    ASSERT(newDistance < averageDistToCenter);
    averageDistToCenter = newDistance;
  }
}
