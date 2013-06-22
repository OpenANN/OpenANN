#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/util/AssertionMacros.h>

namespace OpenANN
{

void Optimizable::errorGradient(int n, double& value, Eigen::VectorXd& grad)
{
  value = error(n);
  grad = gradient(n);
}

void Optimizable::errorGradient(double& value, Eigen::VectorXd& grad)
{
  const int N = examples();
  double tempValue;
  Eigen::VectorXd tempGrad(dimension());
  value = 0.0;
  grad.setZero();
  for(int n = 0; n < N; n++)
  {
    errorGradient(n, tempValue, tempGrad);
    value += tempValue;
    grad += tempGrad;
  }
}
Eigen::VectorXd Optimizable::error(std::vector<int>::const_iterator startN,
                                   std::vector<int>::const_iterator endN)
{
  Eigen::VectorXd errors(endN - startN);
  int n = 0;
  for(std::vector<int>::const_iterator it = startN; it != endN; ++it, ++n)
    errors(n) = error(*it);
  return errors;
}
Eigen::VectorXd Optimizable::gradient(std::vector<int>::const_iterator startN,
                                      std::vector<int>::const_iterator endN)
{
  Eigen::VectorXd g(dimension());
  g.setZero();
  for(std::vector<int>::const_iterator it = startN; it != endN; ++it)
    g += gradient(*it);
  return g;
}

void Optimizable::errorGradient(std::vector<int>::const_iterator startN,
                                std::vector<int>::const_iterator endN,
                                double& value, Eigen::VectorXd& grad)
{
  value = 0.0;
  grad.setZero();
  for(std::vector<int>::const_iterator it = startN; it != endN; ++it)
  {
    OPENANN_CHECK_WITHIN(*it, 0, examples());
    value += error(*it);
    grad += gradient(*it);
  }
}

}
