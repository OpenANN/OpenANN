#pragma once

namespace OpenANN {

/**
 * The stop criteria for optimization algorithms. Note that not every
 * algorithm might check all criteria.
 */
class StopCriteria
{
public:
  static StopCriteria defaultValue;

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
  StopCriteria();
};

}
