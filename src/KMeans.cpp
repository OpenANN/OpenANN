#include <OpenANN/KMeans.h>
#include <limits>

namespace OpenANN
{

KMeans::KMeans(int D, int K)
  : D(D), K(K), C(K, D), v(K), initialized(false)
{
}

Transformer& KMeans::fit(const Eigen::MatrixXd& X)
{
  OPENANN_CHECK_EQUALS(X.cols(), D);

  if(!initialized)
    initialize(X);

  OPENANN_CHECK_EQUALS(X.rows(), clusterIndices.size());

  findClusters(X);
  updateCenters(X);
  return *this;
}

Eigen::MatrixXd KMeans::operator()(const Eigen::MatrixXd& X)
{
  const int N = X.rows();
  Eigen::MatrixXd Y(N, K);
  for(int n = 0; n < N; ++n)
    for(int k = 0; k < K; k++)
      Y(n, k) = (X.row(n) - C.row(k)).norm();
  return Y;
}

Eigen::MatrixXd KMeans::getCenters()
{
  return C;
}

void KMeans::initialize(const Eigen::MatrixXd& X)
{
  rng.generateIndices(X.rows(), clusterIndices, false);
  for(int k = 0; k < K; ++k)
    C.row(k) = X.row(clusterIndices[k]);
  v.setZero();
  initialized = true;
}

void KMeans::findClusters(const Eigen::MatrixXd& X)
{
  const int N = X.rows();
  for(int n = 0; n < N; ++n)
  {
    int cluster = 0;
    double shortedDistance = std::numeric_limits<double>::max();
    for(int k = 0; k < K; k++)
    {
      const double distance = (C.row(k) - X.row(n)).squaredNorm();
      if(distance < shortedDistance)
      {
        cluster = k;
        shortedDistance = distance;
      }
    }
    clusterIndices[n] = cluster;
  }
}

void KMeans::updateCenters(const Eigen::MatrixXd& X)
{
  const int N = X.rows();
  for(int n = 0; n < N; ++n)
  {
    const int cluster = clusterIndices[n];
    v(cluster)++;
    const double eta = 1.0 / (double) v(cluster);
    C.row(cluster) += eta * (X.row(n) - C.row(cluster));
  }
}

} // namespace OpenANN
