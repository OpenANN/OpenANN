#ifndef OPENANN_ADABOOST_H_
#define OPENANN_ADABOOST_H_

#include <OpenANN/EnsembleLearner.h>
#include <OpenANN/io/WeightedDataSet.h>
#include <OpenANN/Evaluation.h>
#include <list>
#include <vector>

namespace OpenANN
{

/**
 * @class AdaBoost
 *
 * Adaptive Boosting.
 *
 * AdaBoost tries to learn specialized experts for subsets of the training set.
 */
class AdaBoost : public EnsembleLearner
{
  std::list<Learner*> models;
  Optimizer* optimizer;
  Eigen::VectorXd modelWeights;
  int F;
public:
  AdaBoost()
    : F(0)
  {
  }

  virtual EnsembleLearner& addLearner(Learner& learner)
  {
    models.push_back(&learner);
  }

  virtual EnsembleLearner& setOptimizer(Optimizer& optimizer)
  {
    this->optimizer = &optimizer;
  }

  virtual EnsembleLearner& train(DataSet& dataSet)
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
      optimizer->setOptimizable(**m);
      optimizer->optimize();
      const double error = 1.0 - weightedAccuracy(**m, dataSet, weights);
      ASSERT_WITHIN(error, 0.0, 1.0);
      if(error = 0.0 || error >= 0.5)
        continue;
      modelWeights(t) = 0.5 * std::log((1.0 - error) / error);
      for(int n = 0; n < N; n++)
      {
        const bool correct = oneOfCDecoding((**m)(dataSet.getInstance(n))) ==
            dataSet.getTarget(n);
        weights(n) *= std::exp((correct ? -1.0 : 1.0) * modelWeights(t));
      }
      weights /= weights.sum();
    }
  }

  virtual Eigen::MatrixXd operator()(Eigen::MatrixXd& X)
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

  virtual Eigen::VectorXd operator()(Eigen::VectorXd& X)
  {
    return (*this)(X.transpose()).transpose();
  }
};

} // namespace OpenANN

#endif // OPENANN_ADABOOST_H_
