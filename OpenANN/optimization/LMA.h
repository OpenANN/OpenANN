#ifndef OPENANN_LMA_H
#define OPENANN_LMA_H

#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <Eigen/Dense>
#include <optimization.h>

namespace OpenANN {

/**
 * @class LMA
 *
 * Levenberg-Marquardt Algorithm.
 *
 * This algorithm works especially well for least squares optimization. The
 * optimization will stop if one of the following stopping criteria is
 * satisfied:
 *  - \f$ |g| < \f$ stop.minimalSearchSpaceStep
 *  - \f$ |E^{t+1}-E^{t}| \leq \f$ stop.minimalValueDifferences
 *    \f$ \cdot max\{|E^{t+1}|,|E^{t}|,1\} \f$
 *  - \f$ t > \f$ stop.maximalIterations
 *
 * The optimization is cubic in each iteration, i. e. the complexity is
 * \f$ O(L^3) \f$, where L is the number of parameters to optimize. In
 * addition, the jacobian matrix is required for this algorithm, that is the
 * required space is in \f$ O(L N) \f$, where N is the size of the training
 * set. However, LMA requires only a few iterations to converge. Thus, it
 * works very well up to a few thousand parameters and a few thousand training
 * examples. In summary, the requirements are:
 * - sum of squared error (SSE)
 * - maximum of a few thousand parameters
 * - maximum of a few thousand training examples
 *
 * [1] Kenneth Levenberg:
 * A Method for the Solution of Certain Problems in Least Squares,
 * Quarterly of Applied Mathematics 2, pp. 164-168, 1944.
 *
 * [2] Donald Marquardt:
 * An Algorithm for Least-Squares Estimation of Nonlinear Parameters,
 * Journal of the Society for Industrial and Applied Mathematics 11 (2),
 * pp. 431-441, 1963.
 */
class LMA : public Optimizer
{
  StoppingCriteria stop;
  Optimizable* opt; // do not delete
  Eigen::VectorXd optimum;
  int iteration, n;
  alglib_impl::ae_state envState;
  Eigen::VectorXd parameters, errorValues;
  Eigen::MatrixXd jacobian;
  alglib::real_1d_array xIn;
  alglib::minlmstate state;
  alglib::minlmreport report;
public:
  LMA();
  virtual ~LMA();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);
  virtual void optimize();
  virtual bool step();
  virtual Eigen::VectorXd result();
  virtual std::string name();
private:
  void initialize();
  void reset();
};

}

#endif // OPENANN_LMA_H
