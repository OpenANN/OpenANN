#ifndef STOPPINGCRITERIA_H
#define STOPPINGCRITERIA_H

namespace OpenANN {

/**
 * @class StoppingCriteria
 *
 * Stopping criteria for optimization algorithms.
 *
 * Note that not every algorithm might check all criteria.
 */
class StoppingCriteria
{
public:
  static StoppingCriteria defaultValue;

  /**
   * Maximal number of objective function evaluations. In backpropagation
   * based optimization the objective function is the error function.
   */
  int maximalFunctionEvaluations;
  /**
   * Maximal number of optimization algorithm iterations.
   */
  int maximalIterations;
  /**
   * Maximal number of optimization algorithm restarts.
   */
  int maximalRestarts;
  /**
   * Minimal objective function value.
   */
  fpt minimalValue;
  /**
   * Minimal objective function difference between iterations.
   */
  fpt minimalValueDifferences;
  /**
   * Minimal step size in the search step (e. g. gradient norm).
   */
  fpt minimalSearchSpaceStep;

  /**
   * Create default stop criteria.
   */
  StoppingCriteria();
};

}

#endif // STOPPINGCRITERIA_H
