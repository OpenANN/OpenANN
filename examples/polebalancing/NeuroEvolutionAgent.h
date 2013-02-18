#pragma once

#include <rl/Agent.h>
#include <DeepNetwork.h>
#include <optimization/IPOPCMAES.h>
#include "DoubleExponentialSmoothing.h"
#include <vector>
#include <list>

using namespace OpenANN;

class NeuroEvolutionAgent : public Agent, public Optimizable
{
  //! Number of hidden nodes.
  int h;
  //! Bias?
  bool b;
  //! Activation function: tanh or linear.
  const std::string a;
  //! Compress weights?
  bool compress;
  //! Number of compression parameters in the hidden layer.
  int m;
  //! With derivatives?
  bool fullyObservable;
  //! Use alpha beta filters to estimate missing state variables?
  bool alphaBetaFilter;
  //! Use double exponential smoothing to estimate missing state variables?
  bool doubleExponentialSmoothing;
  //! Use Gruau's fitness function.
  bool gruauFitness;

  Environment* environment;
  int inputSize;
  DeepNetwork policy;
  IPOPCMAES opt;
  Vt lastState;
  bool firstStep;
  std::vector<DoubleExponentialSmoothing> des;
  std::list<Vt> inputBuffer;
public:
  NeuroEvolutionAgent(int h, bool b, const std::string a,
                      bool compress = false, int m = 0,
                      bool fullyObservable = true,
                      bool alphaBetaFilter = false,
                      bool doubleExponentialSmoothing = false);
  ~NeuroEvolutionAgent();
  virtual void abandoneIn(Environment& environment);
  virtual void chooseAction();
  virtual void chooseOptimalAction();

  virtual Vt currentParameters();
  virtual unsigned int dimension();
  virtual fpt error();
  virtual Vt gradient();
  virtual Mt hessian();
  virtual void initialize();
  virtual bool providesGradient();
  virtual bool providesHessian();
  virtual bool providesInitialization();
  virtual void setParameters(const Vt& parameters);
  void setSigma0(fpt sigma0);
};
