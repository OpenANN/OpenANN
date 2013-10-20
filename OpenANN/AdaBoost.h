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
 * AdaBoost tries to learn specialized experts for subsets of the training set
 * [1]. This implementation can only be used for classification.
 *
 * [1] Y. Freund, R. E. Schapire:
 * A Decision-Theoretic Generalization of on-Line Learning and an Application
 * to Boosting,
 * Journal of Computer and System Sciences 55, pp. 119-139, 1995.
 */
class AdaBoost : public EnsembleLearner
{
  std::list<Learner*> models;
  Optimizer* optimizer;
  Eigen::VectorXd modelWeights;
  int F;
public:
  AdaBoost();
  /**
   * Get weights of the models.
   * @return model weights, sum up to one
   */
  Eigen::VectorXd getWeights();
  virtual EnsembleLearner& addLearner(Learner& learner);
  virtual EnsembleLearner& setOptimizer(Optimizer& optimizer);
  virtual EnsembleLearner& train(DataSet& dataSet);
  virtual Eigen::MatrixXd operator()(Eigen::MatrixXd& X);
  virtual Eigen::VectorXd operator()(Eigen::VectorXd& x);
};

} // namespace OpenANN

#endif // OPENANN_ADABOOST_H_
