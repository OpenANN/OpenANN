#ifndef OPENANN_ADABOOST_H_
#define OPENANN_ADABOOST_H_

#include <OpenANN/EnsembleLearner.h>
#include <list>

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
  AdaBoost();
  virtual EnsembleLearner& addLearner(Learner& learner);
  virtual EnsembleLearner& setOptimizer(Optimizer& optimizer);
  virtual EnsembleLearner& train(DataSet& dataSet);
  virtual Eigen::MatrixXd operator()(Eigen::MatrixXd& X);
  virtual Eigen::VectorXd operator()(Eigen::VectorXd& x);
};

} // namespace OpenANN

#endif // OPENANN_ADABOOST_H_
