#pragma once

#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <Eigen/Dense>

namespace OpenANN {

/**
 * @class CG
 *
 * Conjugate Gradient.
 */
class CG : public Optimizer
{
  StoppingCriteria stop;
  Optimizable* opt; // do not delete
  Eigen::VectorXd optimum;

public:
  CG();
  ~CG();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);
  virtual bool step() {} // TODO
  virtual void optimize();
  virtual Eigen::VectorXd result();
  virtual std::string name();
};

}
