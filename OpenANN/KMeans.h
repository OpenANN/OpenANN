#ifndef OPENANN_KMEANS_H_
#define OPENANN_KMEANS_H_

#include <OpenANN/util/Random.h>
#include <Eigen/Dense>
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
  /**
   * Create KMeans object.
   * @param D number of features
   * @param K number of centers
   */
  KMeans(int D, int K);

  /**
   * Sequential update.
   * @param X new training data, note that the number of samples per update
   *          must never change!
   */
  void update(const Eigen::MatrixXd& X);

  /**
   * Compute for each instance the distances to the centers.
   * @param X each row represents an instance
   * @return each row contains the distances to all centers
   */
  Eigen::MatrixXd operator()(const Eigen::MatrixXd& X);

  /**
   * Get the learned centers.
   * @return each row represents a center
   */
  Eigen::MatrixXd getCenters();

private:
  void initialize(const Eigen::MatrixXd& X);
  void findClusters(const Eigen::MatrixXd& X);
  void updateCenters(const Eigen::MatrixXd& X);
};

} // namespace OpenANN

#endif // OPENANN_KMEANS_H_
