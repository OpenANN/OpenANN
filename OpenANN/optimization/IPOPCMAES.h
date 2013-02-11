#pragma once

#include "Optimizer.h"
#include <io/Logger.h>
#include <optimization/StoppingCriteria.h>
#include <Eigen/Dense>

template<typename T> class CMAES;
template<typename T> class Parameters;

namespace OpenANN {

/**
 * Evolution Strategies with Covariance Matrix Adaption and a restart strategy
 * that increases the population size (IPOP-CMA-ES).
 */
class IPOPCMAES : public Optimizer
{
  Logger logger;
  StoppingCriteria stop;
  bool maxFunEvalsActive;
  Optimizable* opt; // do not delete
  CMAES<fpt>* cmaes;
  Parameters<fpt>* parameters;

  int currentIndividual;
  fpt* initialX;
  fpt* initialStdDev;
  fpt* const* population;
  fpt* fitnessValues;
  int restarts;
  int evaluations;
  int evaluationsAfterRestart;
  bool stopped;

  Vt optimum;
  fpt optimumValue;

  fpt sigma0;

public:
  IPOPCMAES();
  virtual ~IPOPCMAES();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);
  bool restart();
  virtual void optimize();
  virtual bool step();
  Vt getNext();
  void setError(fpt fitness);
  bool terminated();
  virtual Vt result();
  virtual std::string name();
  void setSigma0(fpt sigma0);
};

}
