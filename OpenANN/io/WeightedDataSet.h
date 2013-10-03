#ifndef OPENANN_WEIGHTED_DATA_SET_H_
#define OPENANN_WEIGHTED_DATA_SET_H_

#include <OpenANN/io/DataSet.h>
#include <vector>

namespace OpenANN
{

/**
 * @class WeightedDataSet
 *
 * Resampled dataset based on the original dataset.
 *
 * The probability of each instance to occur in the dataset is defined by the
 * given weights. Note that the weights must sum up to one.
 */
class WeightedDataSet : public DataSet
{
  DataSet& dataSet;
  Eigen::VectorXd weights;
  bool deterministic;
  std::vector<int> originalIndices;
public:
  /**
   * @param dataSet original dataset
   * @param weights weights for each instance, must sum up to one
   * @param deterministic use deterministic (roulette wheel) sampling
   */
  WeightedDataSet(DataSet& dataSet, const Eigen::VectorXd& weights,
                  bool deterministic);
  WeightedDataSet& updateWeights(const Eigen::VectorXd& weights);
  virtual int samples();
  virtual int inputs();
  virtual int outputs();
  virtual Eigen::VectorXd& getInstance(int n);
  virtual Eigen::VectorXd& getTarget(int n);
  virtual void finishIteration(Learner& learner) {}
private:
  void resample();
};

} // namespace OpenANN

#endif // OPENANN_WEIGHTED_DATA_SET_H_
