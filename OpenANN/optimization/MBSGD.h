#pragma once

#include <optimization/Optimizer.h>
#include <optimization/StoppingCriteria.h>
#include <io/Logger.h>
#include <Eigen/Dense>

namespace OpenANN {

/**
 * @class MBSGD
 *
 * Mini-Batch Stochastic Gradient Descent.
 * Some tricks are used to speed up the optimization:
 *  * momentum
 *  * adaptive learning rates per parameter
 */
class MBSGD : public Optimizer
{
  Logger debugLogger;
  StoppingCriteria stop;
  Optimizable* opt; // do not delete
  Vt optimum;
  //! Typical size of a mini-batch is 10 to a few hundred.
  int batchSize;
  fpt alpha, eta, minGain, maxGain;

public:
  MBSGD();
  ~MBSGD();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);
  virtual void optimize();
  virtual bool step();
  virtual Vt result();
  virtual std::string name();
};

}
