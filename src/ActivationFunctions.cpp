#include <OpenANN/ActivationFunctions.h>
#include <limits>
#include <cmath>

namespace OpenANN
{

void activationFunction(ActivationFunction act, const Eigen::VectorXd& a, Eigen::VectorXd& z)
{
  switch(act)
  {
    case LOGISTIC:
      logistic(a, z);
      break;
    case TANH:
      normaltanh(a, z);
      break;
    case TANH_SCALED:
      scaledtanh(a, z);
      break;
    case RECTIFIER:
      rectifier(a, z);
      break;
    case LINEAR:
    default:
      linear(a, z);
      break;
  }
}

void activationFunctionDerivative(ActivationFunction act, const Eigen::VectorXd& z, Eigen::VectorXd& gd)
{
  switch(act)
  {
    case LOGISTIC:
      logisticDerivative(z, gd);
      break;
    case TANH:
      normaltanhDerivative(z, gd);
      break;
    case TANH_SCALED:
      scaledtanhDerivative(z, gd);
      break;
    case RECTIFIER:
      rectifierDerivative(z, gd);
      break;
    case LINEAR:
    default:
      linearDerivative(gd);
      break;
  }
}

void softmax(Eigen::VectorXd& y)
{
  const int F = y.rows();
  const double max = y.maxCoeff();
  for(int f = 0; f < F; f++)
    y(f) = std::exp(y(f) - max);
  y /= y.sum();
}

void logistic(const Eigen::VectorXd& a, Eigen::VectorXd& z)
{
  for(int j = 0; j < a.rows(); j++)
  {
    if(a(j) < -45.0)
      z(j) = 0.0;
    else if(a(j) > 45.0)
      z(j) = 1.0;
    else
      z(j) = 1.0 / (1.0+std::exp(-a(j)));
  }
}

void logisticDerivative(const Eigen::VectorXd& z, Eigen::VectorXd& gd)
{
  for(int j = 0; j < gd.rows(); j++)
    gd(j) = z(j)*(1.0 - z(j));
}

void normaltanh(const Eigen::VectorXd& a, Eigen::VectorXd& z)
{
  for(int j = 0; j < a.rows(); j++)
    z(j) = std::tanh(a(j));
}

void normaltanhDerivative(const Eigen::VectorXd& z, Eigen::VectorXd& gd)
{
  for(int j = 0; j < gd.rows(); j++)
    gd(j) = 1.0 - z(j)*z(j);
}

void scaledtanh(const Eigen::VectorXd& a, Eigen::VectorXd& z)
{
  for(int j = 0; j < a.rows(); j++)
    z(j) = 1.7159*std::tanh(0.66666667*a(j));
}

void scaledtanhDerivative(const Eigen::VectorXd& z, Eigen::VectorXd& gd)
{
  for(int j = 0; j < gd.rows(); j++)
    gd(j) = 0.66666667/1.7159*(1.7159+z(j))*(1.7159-z(j));
}

void rectifier(const Eigen::VectorXd& a, Eigen::VectorXd& z)
{
  for(int j = 0; j < a.rows(); j++)
    z(j) = std::max<double>((double) 0.0, a(j));
}

void rectifierDerivative(const Eigen::VectorXd& z, Eigen::VectorXd& gd)
{
  for(int j = 0; j < gd.rows(); j++)
    gd(j) = z(j) == (double) 0.0 ? (double) 0.0 : (double) 1.0;
}

void linear(const Eigen::VectorXd& a, Eigen::VectorXd& z)
{
  z.middleRows(0, a.rows()) = a;
}

void linearDerivative(Eigen::VectorXd& gd)
{
  gd.fill(1.0);
}

}
