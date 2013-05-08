#include "FiniteDifferences.h"

namespace OpenANN {

namespace FiniteDifferences
{

Eigen::VectorXd inputGradient(const Eigen::VectorXd& x,
                              const Eigen::VectorXd& y, Learner& learner,
                              const double eps)
{
  const int D = x.rows();
  Eigen::VectorXd gradient(D);
  gradient.fill(0.0);
  Eigen::MatrixXd in = x.transpose();
  Eigen::MatrixXd out = y.transpose();
  for(unsigned i = 0; i < D; i++)
  {
    in(i) += eps;
    learner.trainingSet(in, out);
    double errorPlusEps = learner.error(0);

    in(i) -= 2*eps;
    learner.trainingSet(in, out);
    double errorMinusEps = learner.error(0);

    in(i) = x(i);
    gradient(i) = (errorPlusEps - errorMinusEps) / (2.0 * eps);
  }
  return gradient;
}

Eigen::VectorXd parameterGradient(int n, Optimizable& opt, const double eps)
{
  Eigen::VectorXd gradient(opt.dimension());
  gradient.fill(0.0);
  Eigen::VectorXd params = opt.currentParameters();
  Eigen::VectorXd modifiedParams = params;
  for(unsigned i = 0; i < opt.dimension(); i++)
  {
    modifiedParams(i) += eps;
    opt.setParameters(modifiedParams);
    double errorPlusEps = opt.error(n);
    modifiedParams = params;

    modifiedParams(i) -= eps;
    opt.setParameters(modifiedParams);
    double errorMinusEps = opt.error(n);
    modifiedParams = params;

    gradient(i) = (errorPlusEps - errorMinusEps) / (2.0 * eps);
  }
  opt.setParameters(params);
  return gradient;
}

}

}
