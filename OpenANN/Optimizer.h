#pragma once

#include <Eigen/Dense>
#include <string>

namespace OpenANN {

class Optimizable;
class StopCriteria;

/**
 * The common interface of all optimization algorithms.
 */
class Optimizer
{
public:
  virtual ~Optimizer() {}
  virtual void setOptimizable(Optimizable& optimizable) = 0;
  virtual void setStopCriteria(const StopCriteria& sc) = 0;
  virtual void optimize() = 0;
  virtual Vt result() = 0;
  virtual std::string name() = 0;
};

}
