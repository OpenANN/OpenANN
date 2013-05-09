#ifndef OPTIMIZABLE_H
#define OPTIMIZABLE_H
#if __GNUC__ >= 4
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <Eigen/Dense>

namespace OpenANN {

/**
 * @class Optimizable
 *
 * Represents an optimizable object. E. g. objective function, neural network,
 * etc.
 */
class Optimizable
{
public:
  virtual ~Optimizable() {}
  /**
   * @return does the optimizable object provide a parameter initialization?
   */
  virtual bool providesInitialization() = 0;
  /**
   * Initialize the object's parameters.
   */
  virtual void initialize() = 0;
  /**
   * @return number of optimizable parameters
   */
  virtual unsigned dimension() = 0;
  /**
   * @return current parameters
   */
  virtual Eigen::VectorXd currentParameters() = 0;
  /**
   * Set new parameters.
   * @param parameters new parameters
   */
  virtual void setParameters(const Eigen::VectorXd& parameters) = 0;
  /**
   * Compute error on training set.
   * @return current error on training set or objective function value
   */
  virtual double error() = 0;
  /**
   * @return does the optimizable provide a gradient?
   */
  virtual bool providesGradient() = 0;
  /**
   * @return gradient of the objective function with respect to parameters
   */
  virtual Eigen::VectorXd gradient() = 0;
  /**
   * @return does the optimizable provide a hessian?
   */
  virtual bool providesHessian() = 0;
  /**
   * @return hessian of the objective function with respect to parameters
   */
  virtual Eigen::MatrixXd hessian() = 0;
  /**
   * @return number of training examples
   */
  virtual unsigned examples() { return 1; }
  /**
   * @return error of the i-th training example
   */
  virtual double error(unsigned n) { return error(); }
  /**
   * @return gradient of the n-th training example
   */
  virtual Eigen::VectorXd gradient(unsigned n) { return gradient(); }
  /**
   * @return hessian of the n-th training example
   */
  virtual Eigen::VectorXd hessian(unsigned n) { return hessian(); }
  /**
   * Calculates the function value and gradient of a training example.
   * @param n index of training example
   * @param value function value
   * @param grad gradient of the function
   */
  virtual void errorGradient(int n, double& value, Eigen::VectorXd& grad)
  {
    value = error(n);
    grad = gradient(n);
  }
  /**
   * Calculates the function value and gradient of all training examples.
   * @param value function value
   * @param grad gradient of the function
   */
  virtual void errorGradient(double& value, Eigen::VectorXd& grad)
  {
    const int N = examples();
    double tempValue;
    Eigen::VectorXd tempGrad(dimension());
    value = 0.0;
    grad.fill(0.0);
    for(int n = 0; n < N; n++)
    {
      errorGradient(n, tempValue, tempGrad);
      value += tempValue;
      grad += tempGrad;
    }
  }
  /**
   * This callback is called after each optimization algorithm iteration.
   */
  virtual void finishedIteration() {}
};

}

#endif // OPTIMIZABLE_H
