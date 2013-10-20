#ifndef OPENANN_BAGGING_H_
#define OPENANN_BAGGING_H_

#include <OpenANN/EnsembleLearner.h>
#include <list>

namespace OpenANN
{

/**
 * @class Bagging
 *
 * Bootstrap Aggregating.
 *
 * Bagging averages instable learners that have been trained on randomly
 * sampled subsets of the training set [1]. This implementation can be used
 * for classification and regression.
 *
 * [1] L. Breiman: Bagging Predictors, Machine Learning 24, pp. 123-140, 1996.
 */
class Bagging : public EnsembleLearner
{
  std::list<Learner*> models;
  Optimizer* optimizer;
  double bagSize;
  int F;
public:
  Bagging(double bagSize);
  virtual EnsembleLearner& addLearner(Learner& learner);
  virtual EnsembleLearner& setOptimizer(Optimizer& optimizer);
  virtual EnsembleLearner& train(DataSet& dataSet);
  virtual Eigen::MatrixXd operator()(Eigen::MatrixXd& X);
  virtual Eigen::VectorXd operator()(Eigen::VectorXd& x);
};

} // namespace OpenANN

#endif // OPENANN_BAGGING_H_
