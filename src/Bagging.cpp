#include <OpenANN/Bagging.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/io/DataSetView.h>

namespace OpenANN
{

Bagging::Bagging(double bagSize)
  : bagSize(bagSize), F(0)
{
}

EnsembleLearner& Bagging::addLearner(Learner& learner)
{
  models.push_back(&learner);
}

EnsembleLearner& Bagging::setOptimizer(Optimizer& optimizer)
{
  this->optimizer = &optimizer;
}

EnsembleLearner& Bagging::train(DataSet& dataSet)
{
  const int N = dataSet.samples();
  F = dataSet.outputs();
  for(std::list<Learner*>::iterator m = models.begin(); m != models.end(); m++)
  {
    std::vector<DataSetView> dataSets;
    split(dataSets, dataSet, bagSize, true);
    (*m)->trainingSet(dataSets[0]);
    optimizer->setOptimizable(**m);
    optimizer->optimize();
  }
}

Eigen::MatrixXd Bagging::operator()(Eigen::MatrixXd& X)
{
  const int N = X.rows();
  Eigen::MatrixXd Y(N, F);
  Y.fill(0.0);

  for(std::list<Learner*>::iterator m = models.begin(); m != models.end(); m++)
    Y += (**m)(X);

  return Y / models.size();
}

Eigen::VectorXd Bagging::operator()(Eigen::VectorXd& x)
{
  Eigen::MatrixXd X = X.transpose();
  return (*this)(X).transpose();
}

}
