#ifndef OPENANN_LMA_H
#define OPENANN_LMA_H

#ifdef USE_GPL_LICENSE

#include <Optimizer.h>
#include <StopCriteria.h>
#include <io/Logger.h>
#include <Eigen/Dense>

namespace OpenANN {

/**
 * Levenberg-Marquardt Algorithm.
 */
class LMA : public Optimizer
{
  Logger debugLogger;
  StopCriteria stop;
  Optimizable* opt; // do not delete
  Vt optimum;
  bool approximateHessian;

public:
  LMA(bool approximateHessian = false);
  virtual ~LMA();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StopCriteria& stop);
  virtual void optimize();
  virtual Vt result();
  virtual std::string name();
};

}

#else // USE_GPL_LICENSE
#warning LMA is only available under GPL license
#endif // USE_GPL_LICENSE

#endif // OPENANN_LMA_H
