#include "NeuroEvolutionAgent.h"
#include <OpenANN/util/AssertionMacros.h>
#include <cmath>

NeuroEvolutionAgent::NeuroEvolutionAgent(int h, bool b, const std::string& a,
                                         bool compress, int m,
                                         bool fullyObservable,
                                         bool alphaBetaFilter,
                                         bool doubleExponentialSmoothing)
  : h(h), b(b), a(a), compress(compress), m(m),
    fullyObservable(fullyObservable), alphaBetaFilter(alphaBetaFilter),
    doubleExponentialSmoothing(doubleExponentialSmoothing),
    gruauFitness(false),
    environment(0), inputSize(-1), firstStep(true)
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
  policy.inputLayer(inputSize, 1, 1);

  if(!fullyObservable)
  {
    if(alphaBetaFilter)
      policy.alphaBetaFilterLayer(environment.deltaT(), 5.0);
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
    policy.outputLayer(environment.actionSpaceDimension(), act, 0.05, b);
  }
  else
  {
    if(compress)
      policy.compressedOutputLayer(environment.actionSpaceDimension(), m, act, std::string("dct"), 0.05, b);
    else
      policy.outputLayer(environment.actionSpaceDimension(), act, 0.05, b);
  }

  StoppingCriteria stop;
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
    double fitness = 0.0;
    if(!gruauFitness)
      fitness = -environment->stepsInEpisode();
    else
    {
      // Gruau's fitness measurement
      const double f1 = environment->stepsInEpisode() / 1000.0;
      double f2;
      if(environment->stepsInEpisode() >= 100)
      {
        double denom = 0.0;
        for(std::list<Eigen::VectorXd>::iterator it = inputBuffer.begin(); it != inputBuffer.end(); ++it)
        {
          denom += std::fabs((*it)(0)); // position on the track
          for(int i = 1; i < inputSize; i += 2)
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
  Eigen::VectorXd input(inputSize);
  if(fullyObservable || alphaBetaFilter)
    input = state;
  else
  {
    if(doubleExponentialSmoothing)
    {
      int in_idx = 0;
      for(int i = 0; i < state.rows(); i++)
      {
        Eigen::VectorXd estimation = des[i](state(i));
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

const Eigen::VectorXd& NeuroEvolutionAgent::currentParameters()
{
  return policy.currentParameters();
}

unsigned int NeuroEvolutionAgent::dimension()
{
  return policy.dimension();
}

double NeuroEvolutionAgent::error()
{
  OPENANN_CHECK(false);
  return 0.0;
}

Eigen::VectorXd NeuroEvolutionAgent::gradient()
{
  OPENANN_CHECK(false);
  return Eigen::VectorXd();
}

void NeuroEvolutionAgent::initialize()
{
  policy.initialize();
}

bool NeuroEvolutionAgent::providesGradient()
{
  return false;
}

bool NeuroEvolutionAgent::providesInitialization()
{
  return true;
}

void NeuroEvolutionAgent::setParameters(const Eigen::VectorXd& parameters)
{
  policy.setParameters(parameters);
}

void NeuroEvolutionAgent::setSigma0(double sigma0)
{
  opt.setSigma0(sigma0);
}
