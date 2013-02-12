#pragma once

#include <optimization/Optimizer.h>
#include <optimization/StoppingCriteria.h>
#include <io/Logger.h>
#include <Random.h>
#include <Eigen/Dense>
#include <vector>
#include <list>

namespace OpenANN {

/**
 * @class MBSGD
 *
 * Mini-Batch Stochastic Gradient Descent.
 * Some tricks are used to speed up the optimization:
 *  * momentum
 *  * adaptive learning rates per parameter
 * When the batch size equals 1, the algorithms degenerates to stochastic
 * gradient descent. When it equals the training set size, the algorithm is
 * like batch gradient descent.
 */
class MBSGD : public Optimizer
{
  Logger debugLogger;
  //! Stopping criteria
  StoppingCriteria stop;
  //! Optimizable problem
  Optimizable* opt; // do not delete
  //! Optimum parameters
  Vt optimum;
  //! Learning rate
  fpt alpha;
  //! Learning rate decay
  fpt alphaDecay;
  //! Minimum learning rat
  fpt minAlpha;
  //! Momentum
  fpt eta;
  //! Momentum gain
  fpt etaGain;
  //! Maximum momentum
  fpt maxEta;
  //! Typical size of a mini-batch is 10 to a few hundred.
  int batchSize;
  //! Minimum gain of the learning rate per parameter, e. g. 0.01, 0.1 or 1
  fpt minGain;
  //! Maximum gain of the learning rate per parameter, e. g. 1, 10 or 100
  fpt maxGain;
  //! Use parameter adaption
  bool useGain;

  int iteration;
  RandomNumberGenerator rng;
  int P, N, batches;
  Vt gradient, gains, parameters, momentum;
  std::vector<std::list<int> > batchAssignment;
public:
  MBSGD(fpt learningRate = 0.7, fpt learningRateDecay = 0.998,
        fpt minimalLearningRate = 0.0001, fpt momentum = 0.5,
        fpt momentumGain = 0.0005, fpt maximalMomentum = 0.99,
        int batchSize = 50, fpt minGain = 0.1, fpt maxGain = 10.0);
  ~MBSGD();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);
  virtual void optimize();
  virtual bool step();
  virtual Vt result();
  virtual std::string name();
private:
  void initialize();
};

}
