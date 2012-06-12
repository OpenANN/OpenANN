#pragma once

#include <Optimizer.h>
#include <StopCriteria.h>
#include <io/Logger.h>
#include <Eigen/Dense>

namespace OpenANN {

/**
 * Conjugate Gradient.
 */
class SGD : public Optimizer
{
  Logger debugLogger;
  StopCriteria stop;
  Optimizable* opt; // do not delete
  Vt optimum;

  bool regularize;
  fpt regularizationCoefficient;

public:
  SGD();
  ~SGD();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StopCriteria& stop);
  virtual void optimize();
  virtual Vt result();
  virtual std::string name();
};

}
