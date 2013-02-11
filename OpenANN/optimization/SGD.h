#pragma once

#include <optimization/Optimizer.h>
#include <optimization/StoppingCriteria.h>
#include <io/Logger.h>
#include <Eigen/Dense>

namespace OpenANN {

/**
 * Stochastic Gradient Descent.
 */
class SGD : public Optimizer
{
  Logger debugLogger;
  StoppingCriteria stop;
  Optimizable* opt; // do not delete
  Vt optimum;

  bool regularize;
  fpt regularizationCoefficient;

public:
  SGD();
  ~SGD();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);
  virtual void optimize();
  virtual Vt result();
  virtual std::string name();
};

}
