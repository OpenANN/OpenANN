#ifndef OPENANN_LMA_H
#define OPENANN_LMA_H

#ifdef USE_GPL_LICENSE

#include <optimization/Optimizer.h>
#include <optimization/StoppingCriteria.h>
#include <io/Logger.h>
#include <Eigen/Dense>

namespace OpenANN {

/**
 * Levenberg-Marquardt Algorithm.
 *
 * This algorithm works especially well for least squares optimization. The
 * optimization will stop if one of the following stopping criteria is
 * satisfied:
 *  - \f$ |g| < \f$ stop.minimalSearchSpaceStep
 *  - \f$ |E^{t+1}-E^{t}| \leq \f$ stop.minimalValueDifferences
 *    \f$ \cdot max\{|E^{t+1}|,|E^{t}|,1\} \f$
 *  - \f$ t > \f$ stop.maximalIterations
 */
class LMA : public Optimizer
{
  Logger debugLogger;
  StoppingCriteria stop;
  Optimizable* opt; // do not delete
  Vt optimum;
  bool approximateHessian;

public:
  LMA(bool approximateHessian = false);
  virtual ~LMA();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);
  virtual void optimize();
  virtual Vt result();
  virtual std::string name();
};

}

#else // USE_GPL_LICENSE
#warning LMA is only available under GPL license
#endif // USE_GPL_LICENSE

#endif // OPENANN_LMA_H
