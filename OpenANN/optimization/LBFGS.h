#ifndef OPENANN_OPTIMIZATION_LBFGS_H_
#define OPENANN_OPTIMIZATION_LBFGS_H_

#include "Optimizer.h"

namespace OpenANN
{

class LBFGS : public Optimizer
{
public:
  LBFGS()
  {
  }
  virtual ~LBFGS();
  virtual void setStopCriteria(const StoppingCriteria& sc);
  virtual void setOptimizable(Optimizable& optimizable);
  virtual void optimize();
  virtual bool step();
  virtual Eigen::VectorXd result();

  virtual std::string name()
  {
    return "L-BFGS";
  }
};

} // namespace OpenANN

#endif // OPENANN_OPTIMIZATION_LBFGS_H_
