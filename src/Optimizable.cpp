#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/util/AssertionMacros.h>

namespace OpenANN {

void Optimizable::VJ(Eigen::VectorXd& values, Eigen::MatrixXd& jacobian)
{
  OPENANN_CHECK_EQUALS(values.rows(), (int) examples());
  OPENANN_CHECK_EQUALS(jacobian.rows(), (int) examples());
  OPENANN_CHECK_EQUALS(jacobian.cols(), (int) dimension());
  for(unsigned n = 0; n < examples(); n++)
  {
    values(n) = error(n);
    jacobian.row(n) = gradient(n);
  }
}

Eigen::VectorXd Optimizable::singleGradientFD(int n, const double eps)
{
  Eigen::VectorXd gradient(dimension(), 1);
  gradient.fill(0.0);
  Eigen::VectorXd params = currentParameters();
  Eigen::VectorXd modifiedParams = params;
  for(unsigned i = 0; i < dimension(); i++)
  {
    modifiedParams(i, 0) += eps;
    setParameters(modifiedParams);
    double errorPlusEps = error(n);
    modifiedParams = params;

    modifiedParams(i, 0) -= eps;
    setParameters(modifiedParams);
    double errorMinusEps = error(n);
    modifiedParams = params;

    gradient(i, 0) = (errorPlusEps - errorMinusEps) / (2.0 * eps);
  }
  setParameters(params);
  return gradient;
}

Eigen::VectorXd Optimizable::gradientFD(const double eps)
{
  Eigen::VectorXd gradient(dimension(), 1);
  gradient.fill(0.0);
  Eigen::VectorXd params = currentParameters();
  Eigen::VectorXd modifiedParams = params;
  for(unsigned i = 0; i < dimension(); i++)
  {
    modifiedParams(i, 0) += eps;
    setParameters(modifiedParams);
    double errorPlusEps = error();
    modifiedParams = params;

    modifiedParams(i, 0) -= eps;
    setParameters(modifiedParams);
    double errorMinusEps = error();
    modifiedParams = params;

    gradient(i, 0) = (errorPlusEps - errorMinusEps) / (2.0 * eps);
  }
  setParameters(params);
  return gradient;
}

Eigen::MatrixXd Optimizable::hessianFD(const double eps)
{
  Eigen::MatrixXd hessian(dimension(), dimension());
  hessian.fill(0.0);
  Eigen::VectorXd params = currentParameters();
  Eigen::VectorXd modifiedParams = params;
  for(unsigned i = 0; i < dimension(); i++)
  {
    for(unsigned j = 0; j <= i; j++)
    {
      modifiedParams(i, 0) += eps;
      modifiedParams(j, 0) += eps;
      setParameters(modifiedParams);
      double plusplus = error();
      modifiedParams = params;

      modifiedParams(i, 0) += eps;
      modifiedParams(j, 0) -= eps;
      setParameters(modifiedParams);
      double plusminus = error();
      modifiedParams = params;

      modifiedParams(i, 0) -= eps;
      modifiedParams(j, 0) += eps;
      setParameters(modifiedParams);
      double minusplus = error();
      modifiedParams = params;

      modifiedParams(i, 0) -= eps;
      modifiedParams(j, 0) -= eps;
      setParameters(modifiedParams);
      double minusminus = error();
      modifiedParams = params;

      hessian(i, j) = (plusplus - plusminus - minusplus + minusminus) / (4.0 * eps*eps);
      if(i != j)
        hessian(j, i) = hessian(i, j);
    }
  }
  setParameters(params);
  return hessian;
}

}
