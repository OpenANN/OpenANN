#ifndef OPENANN_OPTIMIZATION_H_
#define OPENANN_OPTIMIZATION_H_

#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <Eigen/Core>
#include <optimization.h>

namespace OpenANN
{

/**
 * @class CG
 *
 * Conjugate Gradient.
 *
 * The nonlinear conjugate gradient method is a generalization of the
 * conjugate gradient method that finds the minimum of a quadratic function.
 */
class CG : public Optimizer
{
  StoppingCriteria stop;
  Optimizable* opt; // do not delete
  Eigen::VectorXd optimum;
  int iteration, n;
  Eigen::VectorXd parameters, gradient;
  double error;
  alglib_impl::ae_state envState;
  alglib::mincgstate state;
  alglib::real_1d_array xIn;

public:
  CG();
  ~CG();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);
  virtual bool step();
  virtual void optimize();
  virtual Eigen::VectorXd result();
  virtual std::string name();
private:
  void initialize();
  void reset();
};

} // namespace OpenANN

#endif // OPENANN_OPTIMIZATION_H_
