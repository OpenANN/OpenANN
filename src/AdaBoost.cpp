#include <OpenANN/AdaBoost.h>
#include <OpenANN/Evaluation.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/io/WeightedDataSet.h>

namespace OpenANN
{

AdaBoost::AdaBoost()
  : F(0)
{
}

EnsembleLearner& AdaBoost::addLearner(Learner& learner)
{
  models.push_back(&learner);
}

EnsembleLearner& AdaBoost::setOptimizer(Optimizer& optimizer)
{
  this->optimizer = &optimizer;
}

EnsembleLearner& AdaBoost::train(DataSet& dataSet)
{
  const int N = dataSet.samples();
  F = dataSet.outputs();
  modelWeights.conservativeResize(models.size());
  modelWeights.setZero();
  Eigen::VectorXd weights(dataSet.samples());
  weights.fill(1.0 / (double) dataSet.samples());
  WeightedDataSet resampled(dataSet, weights, true);
  int t = 0;
  for(std::list<Learner*>::iterator m = models.begin(); m != models.end();
      m++, t++)
  {
    (*m)->trainingSet(dataSet);
    optimizer->setOptimizable(**m);
    optimizer->optimize();
    const double error = 1.0 - weightedAccuracy(**m, dataSet, weights);
    OPENANN_CHECK_WITHIN(error, 0.0, 1.0);
    if(error == 0.0 || error >= 0.5)
      continue;
    modelWeights(t) = 0.5 * std::log((1.0 - error) / error);
    for(int n = 0; n < N; n++)
    {
      const bool correct = oneOfCDecoding((**m)(dataSet.getInstance(n))) ==
          oneOfCDecoding(dataSet.getTarget(n));
      weights(n) *= std::exp((correct ? -1.0 : 1.0) * modelWeights(t));
    }
    weights /= weights.sum();
  }
}

Eigen::MatrixXd AdaBoost::operator()(Eigen::MatrixXd& X)
{
  const int N = X.rows();
  Eigen::MatrixXd Y(N, F);
  Y.fill(0.0);

  int t = 0;
  for(std::list<Learner*>::iterator m = models.begin(); m != models.end();
      m++, t++)
    Y += modelWeights(t) * (**m)(X);

  return Y;
}

Eigen::VectorXd AdaBoost::operator()(Eigen::VectorXd& x)
{
  Eigen::MatrixXd X = X.transpose();
  return (*this)(X).transpose();
}

}
