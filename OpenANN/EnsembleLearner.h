#ifndef OPENANN_ENSEMBLE_H_
#define OPENANN_ENSEMBLE_H_

#include <OpenANN/Learner.h>
#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/io/DataSet.h>
#include <Eigen/Core>

namespace OpenANN
{

class EnsembleLearner
{
public:
  virtual ~EnsembleLearner() {}
  virtual EnsembleLearner& addLearner(Learner& learner) = 0;
  virtual EnsembleLearner& setOptimizer(Optimizer& optimizer) = 0;
  virtual EnsembleLearner& train(DataSet& dataSet) = 0;
  virtual Eigen::MatrixXd operator()(Eigen::MatrixXd& X) = 0;
  virtual Eigen::VectorXd operator()(Eigen::VectorXd& X) = 0;
};

} // namespace OpenANN

#endif // OPENANN_ENSEMBLE_H_
