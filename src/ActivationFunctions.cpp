#include <ActivationFunctions.h>
#include <limits>
#include <cmath>

namespace OpenANN
{

void softmax(Vt& y)
{
  const int F = y.rows();
  const fpt max = y.maxCoeff();
  for(int f = 0; f < F; f++)
    y(f) = std::exp(y(f) - max);
  y /= y.sum();
}

void logistic(const Vt& a, Vt& z)
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

void logisticDerivative(const Vt& z, Vt& gd)
{
  for(int j = 0; j < gd.rows(); j++)
    gd(j) = z(j)*(1.0 - z(j));
}

void normaltanh(const Vt& a, Vt& z)
{
  for(int j = 0; j < a.rows(); j++)
    z(j) = std::tanh(a(j));
}

void normaltanhDerivative(const Vt& z, Vt& gd)
{
  for(int j = 0; j < gd.rows(); j++)
    gd(j) = 1.0 - z(j)*z(j);
}

void linear(const Vt& a, Vt& z)
{
  z.middleRows(0, a.rows()) = a;
}

void linearDerivative(Vt& gd)
{
  gd.fill(1.0);
}

void scaledtanh(const Vt& a, Vt& z)
{
  for(int j = 0; j < a.rows(); j++)
    z(j) = 1.7159*std::tanh(0.66666667*a(j));
}

void scaledtanhDerivative(const Vt& z, Vt& gd)
{
  for(int j = 0; j < gd.rows(); j++)
    gd(j) = 0.66666667/1.7159*(1.7159+z(j))*(1.7159-z(j));
}

}
