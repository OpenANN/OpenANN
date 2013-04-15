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
  virtual Vt currentParameters() = 0;
  /**
   * Set new parameters.
   * @param parameters new parameters
   */
  virtual void setParameters(const Vt& parameters) = 0;
  /**
   * Compute error on training set.
   * @return current error on training set or objective function value
   */
  virtual fpt error() = 0;
  /**
   * @return does the optimizable provide a gradient?
   */
  virtual bool providesGradient() = 0;
  /**
   * @return gradient of the objective function with respect to parameters
   */
  virtual Vt gradient() = 0;
  /**
   * @return does the optimizable provide a hessian?
   */
  virtual bool providesHessian() = 0;
  /**
   * @return hessian of the objective function with respect to parameters
   */
  virtual Mt hessian() = 0;
  /**
   * @return number of training examples
   */
  virtual unsigned examples() { return 1; }
  /**
   * @return error of the i-th training example
   */
  virtual fpt error(unsigned i) { return error(); }
  /**
   * @return gradient of the i-th training example
   */
  virtual Vt gradient(unsigned i) { return gradient(); }
  /**
   * @return hessian of the i-th training example
   */
  virtual Vt hessian(unsigned i) { return hessian(); }
  /**
   * Calculates the function values and gradients of all training examples.
   * @param values function values
   * @param jacobian contains one gradient per row
   */
  virtual void VJ(Vt& values, Mt& jacobian);
  /**
   * This callback is called after each optimization algorithm iteration.
   */
  virtual void finishedIteration() {}
  /**
   * Use finite differences to approximate the gradient.
   * @return approximated gradient of the n-th training example
   */
  virtual Vt singleGradientFD(int n, const fpt eps = (fpt) 1e-2);
  /**
   * Use finite differences to approximate the gradient.
   * @return approximated gradient of the training set
   */
  virtual Vt gradientFD(const fpt eps = (fpt) 1e-2);
  /**
   * Use finite differences to approximate the hessian.
   * @return approximated hessian of the training set
   */
  virtual Mt hessianFD(const fpt eps = (fpt) 1e-2);
};

}

#endif // OPTIMIZABLE_H
