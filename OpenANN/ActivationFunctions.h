#pragma once

#include <Eigen/Dense>

namespace OpenANN
{

enum ActivationFunction
{
  LOGISTIC,
  TANH,
  TANH_SCALED,
  RECTIFIER,
  LINEAR
};

void activationFunction(ActivationFunction act, const Vt& a, Vt& z);
void activationFunctionDerivative(ActivationFunction act, const Vt& z, Vt& gd);

void softmax(Vt& y);
void logistic(const Vt& a, Vt& z);
void logisticDerivative(const Vt& z, Vt& gd);
void normaltanh(const Vt& a, Vt& z);
void normaltanhDerivative(const Vt& z, Vt& gd);
void scaledtanh(const Vt& a, Vt& z);
void scaledtanhDerivative(const Vt& z, Vt& gd);
void rectifier(const Vt& a, Vt& z);
void rectifierDerivative(const Vt& z, Vt& gd);
void linear(const Vt& a, Vt& z);
void linearDerivative(Vt& gd);

}
