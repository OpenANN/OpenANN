#ifndef OPENANN_ACTIVATION_FUNCTIONS_H_
#define OPENANN_ACTIVATION_FUNCTIONS_H_

#include <Eigen/Core>

namespace OpenANN
{

enum ActivationFunction
{
  /**
   * Logistic sigmoid activation function.
   *
   * Range: [0, 1].
   *
   * \f$ g(a) = \frac{1}{1+\exp{{-a}}} \f$
   */
  LOGISTIC = 0,
  /**
   * Tanh sigmoid function.
   *
   * Range: [-1, 1].
   *
   * \f$ g(a) = tanh(a) \f$
   */
  TANH = 1,
  /**
   * Scaled tanh sigmoid function.
   *
   * This activation function does not saturate around the output -1 or +1.
   *
   * Range: [-1.7159, 1.7159].
   *
   * \f$ g(a) = 1.7159 tanh(\frac{2}{3} a) \f$
   */
  TANH_SCALED = 2,
  /**
   * Biologically inspired, non-saturating rectified linear unit (ReLU).
   *
   * Range: [0, \f$ \infty \f$].
   *
   * \f$ g(a) = max(0, a) \f$
   *
   * [1] X. Glorot, A. Bordes, Y. Bengio:
   * Deep Sparse Rectifier Neural Networks,
   * International Conference on Artificial Intelligence and Statistics 15,
   * pp. 315–323, 2011.
   */
  RECTIFIER = 3,
  /**
   * Identity function.
   *
   * Note that LINEAR and SOFTMAX are actually the same. Which function will
   * be used actually depends on the error function: if it is MSE we will use
   * the identity, if it is CE we will use softmax.
   *
   * Range: [\f$ -\infty \f$, \f$ \infty \f$].
   *
   * \f$ g(a) = a \f$
   */
  LINEAR = 4,
  /**
   * Softmax activation function.
   *
   * Note that LINEAR and SOFTMAX are actually the same. Which function will
   * be used actually depends on the error function: if it is MSE we will use
   * the identity, if it is CE we will use softmax.
   *
   * Range: [0, 1].
   *
   * \f$ g(a_i) = \frac{\exp(a_i)}{\sum_j exp(a_j)} \f$
   */
  SOFTMAX = 4
};

void activationFunction(ActivationFunction act, const Eigen::MatrixXd& a,
                        Eigen::MatrixXd& z);
void activationFunctionDerivative(ActivationFunction act,
                                  const Eigen::MatrixXd& z,
                                  Eigen::MatrixXd& gd);

void softmax(Eigen::MatrixXd& y);
void logistic(const Eigen::MatrixXd& a, Eigen::MatrixXd& z);
void logisticDerivative(const Eigen::MatrixXd& z, Eigen::MatrixXd& gd);
void normaltanh(const Eigen::MatrixXd& a, Eigen::MatrixXd& z);
void normaltanhDerivative(const Eigen::MatrixXd& z, Eigen::MatrixXd& gd);
void scaledtanh(const Eigen::MatrixXd& a, Eigen::MatrixXd& z);
void scaledtanhDerivative(const Eigen::MatrixXd& z, Eigen::MatrixXd& gd);
void rectifier(const Eigen::MatrixXd& a, Eigen::MatrixXd& z);
void rectifierDerivative(const Eigen::MatrixXd& z, Eigen::MatrixXd& gd);
void linear(const Eigen::MatrixXd& a, Eigen::MatrixXd& z);
void linearDerivative(Eigen::MatrixXd& gd);

} // namespace OpenANN

#endif // OPENANN_ACTIVATION_FUNCTIONS_H_
