#ifndef OPENANN_OPTIMIZATION_IPOPCMAES_H_
#define OPENANN_OPTIMIZATION_IPOPCMAES_H_

#include <OpenANN/io/Logger.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/optimization/Optimizer.h>
#include <Eigen/Dense>

template<typename T> class CMAES;
template<typename T> class Parameters;

namespace OpenANN
{

/**
 * @class IPOPCMAES
 *
 * Evolution Strategies with Covariance Matrix Adaption and a restart strategy
 * that increases the population size (IPOP-CMA-ES).
 *
 * IPOP-CMA-ES is an evolutionary optimization algorithm that requires no
 * gradient. The following stopping criteria will be regarged:
 *
 * - maximalFunctionEvaluations
 * - maximalIterations
 * - minimalValue
 * - minimalValueDifferences
 * - minimalSearchSpaceStep
 * - maximalRestarts
 *
 * IPOPCMAES does not support step-wise execution with step(). Use the
 * functions getNext() and setError() instead to get the next parameter vector
 * and set fitness values respectively.
 *
 * [1] Hansen and Ostermeier:
 * Completely Derandomized Self-Adaptation in Evolution Strategies.
 * Evolutionary Computation, 9 (2), pp. 159-195, 2001.
 *
 * [2] Auger and Hansen:
 * A Restart CMA Evolution Strategy With Increasing Population Size.
 * IEEE Congress on Evolutionary Computation, pp. 1769-1776, 2005.
 */
class IPOPCMAES : public Optimizer
{
  Logger logger;
  StoppingCriteria stop;
  bool maxFunEvalsActive;
  Optimizable* opt; // do not delete
  CMAES<double>* cmaes;
  Parameters<double>* parameters;

  int currentIndividual;
  double* initialX;
  double* initialStdDev;
  double* const* population;
  double* fitnessValues;
  int restarts;
  int evaluations;
  int evaluationsAfterRestart;
  bool stopped;

  Eigen::VectorXd optimum;
  double optimumValue;

  double sigma0;

public:
  /**
   * Create an instance of IPOPCMAES.
   */
  IPOPCMAES();
  virtual ~IPOPCMAES();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);
  /**
   * Restart the optimizer.
   * @return optimizer is still running
   */
  bool restart();
  virtual void optimize();
  virtual bool step();
  /**
   * Get next parameter vector.
   * @return parameter vector
   */
  Eigen::VectorXd getNext();
  /**
   * Set fitness of last individual.
   * @param fitness fitness
   */
  void setError(double fitness);
  /**
   * Did the optimizer finish?
   * @return terminated
   */
  bool terminated();
  virtual Eigen::VectorXd result();
  virtual std::string name();
  /**
   * Set the initial step size.
   * @param sigma0 initial step size
   */
  void setSigma0(double sigma0);
};

} // namespace OpenANN

#endif // OPENANN_OPTIMIZATION_IPOPCMAES_H_
