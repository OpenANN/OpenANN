#include "FiniteDifferences.h"

namespace OpenANN
{

namespace FiniteDifferences
{

Eigen::MatrixXd inputGradient(const Eigen::MatrixXd& X,
                              const Eigen::MatrixXd& Y, Learner& learner,
                              const double eps)
{
  const int N = X.rows();
  const int D = X.cols();
  std::vector<int> indices;
  for(int n = 0; n < N; n++)
    indices.push_back(n);

  Eigen::MatrixXd gradient(N, D);
  gradient.fill(0.0);
  Eigen::MatrixXd in = X;
  Eigen::MatrixXd out = Y;
  for(unsigned i = 0; i < D; i++)
  {
    in.col(i).array() += eps;
    learner.trainingSet(in, out);
    Eigen::VectorXd errorPlusEps = learner.error(indices.begin(), indices.end());

    in.col(i).array() -= 2 * eps;
    learner.trainingSet(in, out);
    Eigen::VectorXd errorMinusEps = learner.error(indices.begin(), indices.end());

    in.col(i) = X.col(i);
    gradient.col(i) = (errorPlusEps - errorMinusEps) / (2.0 * eps);
  }
  return gradient;
}

Eigen::VectorXd parameterGradient(int n, Optimizable& opt, const double eps)
{
  std::vector<int> indices;
  indices.push_back(n);
  return parameterGradient(indices.begin(), indices.end(), opt, eps);
}

Eigen::VectorXd parameterGradient(std::vector<int>::const_iterator start,
                                  std::vector<int>::const_iterator end,
                                  Optimizable& opt, const double eps)
{
  Eigen::VectorXd gradient(opt.dimension());
  gradient.fill(0.0);
  Eigen::VectorXd params = opt.currentParameters();
  Eigen::VectorXd modifiedParams = params;
  for(unsigned i = 0; i < opt.dimension(); i++)
  {
    modifiedParams(i) += eps;
    opt.setParameters(modifiedParams);
    double errorPlusEps = opt.error(start, end).sum();
    modifiedParams = params;

    modifiedParams(i) -= eps;
    opt.setParameters(modifiedParams);
    double errorMinusEps = opt.error(start, end).sum();
    modifiedParams = params;

    gradient(i) += (errorPlusEps - errorMinusEps) / (2.0 * eps);
    opt.setParameters(params);
  }
  return gradient;
}

}

}
