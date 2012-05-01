#pragma once
#if __GNUC__ >= 4
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <Eigen/Dense>

namespace OpenANN {

/**
 * Represents an optimizable object. E. g. objective function, neural network,
 * etc.
 */
class Optimizable
{
public:
  virtual ~Optimizable() {}
  virtual bool providesInitialization() = 0;
  virtual void initialize() = 0;
  /**
   * The number of optimizable parameters.
   */
  virtual unsigned dimension() = 0;
  virtual Vt currentParameters() = 0;
  virtual void setParameters(const Vt& parameters) = 0;
  /**
   * The current error on training set or objective function value.
   */
  virtual fpt error() = 0;
  virtual bool providesGradient() = 0;
  virtual Vt gradient() = 0;
  virtual bool providesHessian() = 0;
  virtual Mt hessian() = 0;
  virtual unsigned examples() { return 1; }
  virtual fpt error(unsigned i) { return error(); }
  virtual Vt gradient(unsigned i) { return gradient(); }
  virtual Vt hessian(unsigned i) { return hessian(); }
  virtual int operator()(const Vt& x, Vt& fvec);
  virtual int df(const Vt& x, Mt& fjac);
  virtual int inputs() { return dimension(); }
  virtual int values() { return examples(); }
  virtual void VJ(Vt& values, Mt& jacobian);
  virtual void finishedIteration() {}
  virtual Vt singleGradientFD(int n, const fpt eps = 1e-10);
  virtual Vt gradientFD(const fpt eps = 1e-10);
  virtual Mt hessianFD(const fpt eps = 1e-5);
};

}
