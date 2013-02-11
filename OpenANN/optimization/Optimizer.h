#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>
#include <string>

namespace OpenANN {

class Optimizable;
class StoppingCriteria;

/**
 * The common interface of all optimization algorithms.
 */
class Optimizer
{
public:
  virtual ~Optimizer() {}
  /**
   * Pass the objective function.
   * @param optimizable objective function, e. g. error function of an ANN
   */
  virtual void setOptimizable(Optimizable& optimizable) = 0;
  /**
   * Pass the stop criteria.
   * @param sc the parameters used to stop the optimization
   */
  virtual void setStopCriteria(const StoppingCriteria& sc) = 0;
  /**
   * Optimize until the optimization meets the stop criteria.
   */
  virtual void optimize() = 0;
  /**
   * Determine the best result.
   * @return the best parameter the algorithm found
   */
  virtual Vt result() = 0;
  /**
   * @return name of the optimization algorithm
   */
  virtual std::string name() = 0;
};

}

#endif // OPTIMIZER_H
