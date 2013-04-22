#include <OpenANN/Evaluation.h>
#include <cmath>

namespace OpenANN {

double sse(Learner& learner, DataSet& dataSet)
{
  const int N = dataSet.samples();
  double sse = 0.0;
  for(int n = 0; n < N; n++)
    sse += (learner(dataSet.getInstance(n)) - dataSet.getTarget(n)).squaredNorm();
  return sse;
}

double mse(Learner& learner, DataSet& dataSet)
{
  return sse(learner, dataSet) / (double) dataSet.samples();
}

double rmse(Learner& learner, DataSet& dataSet)
{
  return std::sqrt(mse(learner, dataSet));
}

double ce(Learner& learner, DataSet& dataSet)
{
  // TODO implement
}


double accuracy(Learner& learner, DataSet& dataSet)
{
  // TODO implement
}

Eigen::MatrixXd confusionMatrix(Learner& learner, DataSet& dataSet)
{
  // TODO implement
}


int oneOfCDecoding(const Eigen::VectorXd& target)
{
  // TODO implement
}

}
