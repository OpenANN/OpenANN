#include "NeuroEvolutionAgent.h"
#include <AssertionMacros.h>
#include <cmath>

NeuroEvolutionAgent::NeuroEvolutionAgent(int h, bool b, const std::string a,
                                         bool compress, int m,
                                         bool fullyObservable,
                                         bool alphaBetaFilter,
                                         bool doubleExponentialSmoothing)
  : h(h), b(b), a(a), compress(compress), m(m),
    fullyObservable(fullyObservable), alphaBetaFilter(alphaBetaFilter),
    doubleExponentialSmoothing(doubleExponentialSmoothing),
    gruauFitness(false), policy(DeepNetwork::SSE)
{
}

NeuroEvolutionAgent::~NeuroEvolutionAgent()
{
}

void NeuroEvolutionAgent::abandoneIn(Environment& environment)
{
  this->environment = &environment;

  ActivationFunction act = a == "tanh" ? TANH : LINEAR;
  inputSize = (fullyObservable || alphaBetaFilter ? 1 : 2)
      * environment.stateSpaceDimension();
  policy.inputLayer(inputSize, 1, 1, b);

  if(!fullyObservable)
  {
    if(alphaBetaFilter)
      policy.alphaBetaFilterLayer(environment.deltaT(), 5.0, b);
    else if(doubleExponentialSmoothing)
    {
      des.resize(environment.stateSpaceDimension());
      for(int i  = 0; i < environment.stateSpaceDimension(); i++)
        des[i].restart();
    }
    else
    {
      lastState.resize(environment.stateSpaceDimension());
      firstStep = true;
    }
  }

  if(h > 0)
  {
    if(compress)
      policy.compressedLayer(h, m, act, std::string("dct"), 0.05, b);
    else
      policy.fullyConnectedLayer(h, act, 0.05, b);
    policy.outputLayer(environment.actionSpaceDimension(), act, 0.05);
  }
  else
  {
    if(compress)
      policy.compressedOutputLayer(environment.actionSpaceDimension(), m, act, std::string("dct"), 0.05);
    else
      policy.outputLayer(environment.actionSpaceDimension(), act, 0.05);
  }

  StopCriteria stop;
  stop.maximalFunctionEvaluations = 1000;
  stop.maximalRestarts = 1000;
  opt.setOptimizable(*this);
  opt.setStopCriteria(stop);
  opt.restart();
  setParameters(opt.getNext());
}

void NeuroEvolutionAgent::chooseAction()
{
  OPENANN_CHECK(environment);

  chooseOptimalAction();

  if(environment->terminalState())
  {
    fpt fitness = 0.0;
    if(!gruauFitness)
      fitness = -environment->stepsInEpisode();
    else
    { // Gruau's fitness measurement
      const fpt f1 = environment->stepsInEpisode() / 1000.0;
      fpt f2;
      if(environment->stepsInEpisode() >= 100)
      {
        fpt denom = 0.0;
        for(std::list<Vt>::iterator it = inputBuffer.begin(); it != inputBuffer.end(); it++)
        {
          denom += std::fabs((*it)(0)); // position on the track
          for(int i = 1; i < inputSize; i+=2)
            denom += std::fabs((*it)(i)); // velocities
        }
        f2 = 0.75 / denom;
      }
      else
        f2 = 0.0;
      fitness = -0.1 * f1 - 0.9 * f2;
    }
    opt.setError(fitness);
    if(opt.terminated())
      opt.restart();
    setParameters(opt.getNext());
    firstStep = true;
    for(size_t i = 0; i < des.size(); i++)
      des[i].restart();
  }
}

void NeuroEvolutionAgent::chooseOptimalAction()
{
  Environment::State state = environment->getState();

  // calculating network input
  Vt input(inputSize);
  if(fullyObservable || alphaBetaFilter)
    input = state;
  else
  {
    if(doubleExponentialSmoothing)
    {
      int in_idx = 0;
      for(int i = 0; i < state.rows(); i++)
      {
        Vt estimation = des[i](state(i));
        input(in_idx++) = estimation(0);
        input(in_idx++) = estimation(1);
      }
    }
    else
    {
      if(firstStep)
      {
        lastState = state;
        firstStep = false;
      }
      int i = 0;
      for(; i < state.rows(); i++)
        input(i) = state(i);
      for(int j = 0; j < state.rows(); i++, j++)
        input(i) = lastState(j);
      lastState = state;
    }
  }

  if(gruauFitness)
  {
    inputBuffer.push_front(input);
    if(inputBuffer.size() > 100)
      inputBuffer.pop_back();
  }

  Environment::Action action = policy(input);
  environment->stateTransition(action);
}

Vt NeuroEvolutionAgent::currentParameters()
{
  return policy.currentParameters();
}

unsigned int NeuroEvolutionAgent::dimension()
{
  return policy.dimension();
}

fpt NeuroEvolutionAgent::error()
{
  OPENANN_CHECK(false);
  return 0.0;
}

Vt NeuroEvolutionAgent::gradient()
{
  OPENANN_CHECK(false);
  return Vt();
}

Mt NeuroEvolutionAgent::hessian()
{
  OPENANN_CHECK(false);
  return Mt();
}

void NeuroEvolutionAgent::initialize()
{
  policy.initialize();
}

bool NeuroEvolutionAgent::providesGradient()
{
  return false;
}

bool NeuroEvolutionAgent::providesHessian()
{
  return false;
}

bool NeuroEvolutionAgent::providesInitialization()
{
  return true;
}

void NeuroEvolutionAgent::setParameters(const Vt& parameters)
{
  policy.setParameters(parameters);
}

void NeuroEvolutionAgent::setSigma0(fpt sigma0)
{
  opt.setSigma0(sigma0);
}
