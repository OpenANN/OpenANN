#include <OpenANN/io/WeightedDataSet.h>
#include <OpenANN/util/Random.h>

namespace OpenANN
{

WeightedDataSet::WeightedDataSet(DataSet& dataSet, const Eigen::VectorXd& weights,
                                 bool deterministic)
  : dataSet(dataSet), weights(weights), deterministic(deterministic)
{
  resample();
}

WeightedDataSet& WeightedDataSet::updateWeights(const Eigen::VectorXd& weights)
{
  this->weights = weights;
  resample();
}

int WeightedDataSet::samples()
{
  return dataSet.samples();
}

int WeightedDataSet::inputs()
{
  return dataSet.inputs();
}

int WeightedDataSet::outputs()
{
  return dataSet.outputs();
}

Eigen::VectorXd& WeightedDataSet::getInstance(int n)
{
  return dataSet.getInstance(originalIndices[n]);
}

Eigen::VectorXd& WeightedDataSet::getTarget(int n)
{
  return dataSet.getTarget(originalIndices[n]);
}

void WeightedDataSet::resample()
{
  const int N = dataSet.samples();
  originalIndices.resize(N);
  RandomNumberGenerator rng;
  for(int n = 0; n < N; n++)
  {
    const double p = deterministic ?
        (double) (n+1) / (double) N : rng.generate<double>(0.0, 1.0);
    double sum = 0.0;
    int idx = 0;
    for(; sum < p && idx < N; idx++)
      sum += weights(idx);
    originalIndices[n] = idx-1;
  }
}

} // namespace OpenANN
