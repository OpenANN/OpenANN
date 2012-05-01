#pragma once

#include <Eigen/Dense>

namespace OpenANN
{

void softmax(Vt& y);
void logistic(const Vt& a, Vt& z);
void logisticDerivative(const Vt& z, Vt& gd);
void normaltanh(const Vt& a, Vt& z);
void normaltanhDerivative(const Vt& z, Vt& gd);
void linear(const Vt& a, Vt& z);
void linearDerivative(Vt& gd);

}
