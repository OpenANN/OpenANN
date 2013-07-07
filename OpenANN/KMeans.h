#ifndef OPENANN_KMEANS_H_
#define OPENANN_KMEANS_H_

#include <OpenANN/util/Random.h>
#include <Eigen/Dense>
#include <limits>
#include <vector>

namespace OpenANN
{

class KMeans
{
  const int D;
  const int K;
  Eigen::MatrixXd C;
  Eigen::VectorXi v;
  bool initialized;
  RandomNumberGenerator rng;
  std::vector<int> clusterIndices;
public:
  KMeans(int D, int K) : D(D), K(K), C(K, D), v(K), initialized(false)
  {
  }

  void update(const Eigen::MatrixXd& X)
  {
    if(!initialized)
      initialize(X);
    findClusters(X);
    updateCenters(X);
  }

  Eigen::MatrixXd operator()(const Eigen::MatrixXd& X)
  {
    const int N = X.rows();
    Eigen::MatrixXd Y(N, K);
    for(int n = 0; n < N; ++n)
      for(int k = 0; k < K; k++)
        Y(n, k) = (X.row(n) - C.row(k)).squaredNorm();
    return Y;
  }

  Eigen::MatrixXd getCenters()
  {
    return C;
  }

private:
  void initialize(const Eigen::MatrixXd& X)
  {
    rng.generateIndices(X.rows(), clusterIndices, false);
    for(int k = 0; k < K; ++k)
      C.row(k) = X.row(clusterIndices[k]);
    v.setZero();
    initialized = true;
  }

  void findClusters(const Eigen::MatrixXd& X)
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

  void updateCenters(const Eigen::MatrixXd& X)
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
};

} // namespace OpenANN

#endif // OPENANN_KMEANS_H_
