#include <optimization/IPOPCMAES.h>
#include <Optimizable.h>
#include <AssertionMacros.h>
#include <cma-es/cmaes.h>

namespace OpenANN {

IPOPCMAES::IPOPCMAES()
    : logger(Logger::NONE),
      maxFunEvalsActive(false),
      opt(0),
      cmaes(0),
      parameters(new Parameters<fpt>),
      currentIndividual(0),
      initialX(0),
      initialStdDev(0),
      population(0),
      fitnessValues(0),
      restarts(-1),
      evaluations(0),
      evaluationsAfterRestart(0),
      stopped(false),
      optimumValue(std::numeric_limits<fpt>::max()),
      sigma0(10.0)
{
}

IPOPCMAES::~IPOPCMAES()
{
  if(cmaes)
  {
    delete cmaes;
    cmaes = 0;
  }
  if(parameters)
  {
    delete parameters;
    parameters = 0;
  }
  if(initialX)
  {
    delete[] initialX;
    initialX = 0;
  }
  if(initialStdDev)
  {
    delete[] initialStdDev;
    initialStdDev = 0;
  }
}

void IPOPCMAES::setOptimizable(Optimizable& opt)
{
  this->opt = &opt;
}

void IPOPCMAES::setStopCriteria(const StoppingCriteria& stop)
{
  OPENANN_CHECK(!cmaes);

  this->stop = stop;
  if(stop.maximalFunctionEvaluations != StoppingCriteria::defaultValue.maximalFunctionEvaluations)
  {
    maxFunEvalsActive = true;
  }
  if(stop.maximalIterations != StoppingCriteria::defaultValue.maximalIterations)
  {
    parameters->stopMaxIter = (fpt) stop.maximalIterations;
  }
  if(stop.minimalValue != StoppingCriteria::defaultValue.minimalValue)
  {
    parameters->stStopFitness.flg = true;
    parameters->stStopFitness.val = stop.minimalValue;
  }
  if(stop.minimalValueDifferences != StoppingCriteria::defaultValue.minimalValueDifferences)
  {
    parameters->stopTolFun = stop.minimalValueDifferences;
  }
  if(stop.minimalSearchSpaceStep != StoppingCriteria::defaultValue.minimalSearchSpaceStep)
  {
    parameters->stopTolX = stop.minimalSearchSpaceStep;
  }
}

bool IPOPCMAES::restart()
{
  OPENANN_CHECK(opt);

  restarts++;
  if(restarts > stop.maximalRestarts)
    return false;

  const unsigned N = opt->dimension();
  if(cmaes)
  {
    evaluations = (int) cmaes->countevals;
    delete cmaes;
  }
  cmaes = new CMAES<fpt>;

  OPENANN_CHECK_WITHIN(restarts, 0, stop.maximalRestarts);
  if(initialX)
    delete[] initialX;
  initialX = new fpt[N];
  if(initialStdDev)
    delete[] initialStdDev;
  initialStdDev = new fpt[N];

  parameters->updateCmode.maxtime = 1.0;
  if(restarts > 0)
    parameters->lambda = (int) ((fpt) parameters->lambda * 2.);

  if(opt->providesInitialization())
  {
    opt->initialize();
    Vt initial = opt->currentParameters();
    for(unsigned i = 0; i < N; i++)
      initialX[i] = initial(i);
  }
  else
  {
    for(unsigned i = 0; i < N; i++)
      initialX[i] = (fpt) i / (fpt) N - (fpt) (N/2);
  }
  for(unsigned i = 0; i < N; i++)
    initialStdDev[i] = sigma0;
    
  parameters->init(N, initialX, initialStdDev);
  fitnessValues = cmaes->init(*parameters);
  cmaes->countevals = (fpt) evaluations;
  evaluationsAfterRestart = 0;

  return true;
}

void IPOPCMAES::optimize()
{
  OPENANN_CHECK(opt);

  while(restart())
  {
    while(!terminated())
    {
      for(int i = 0; i < cmaes->get(CMAES<fpt>::PopSize); ++i)
      {
        Vt individual = getNext();
        opt->setParameters(individual);
        fpt error = opt->error();
        setError(error);
      }
    }

    if(cmaes->get(CMAES<fpt>::FBestEver) < optimumValue)
    {
      // TODO actually XMean should be the best estimator
      optimum.resize(opt->dimension(), 1);
      fpt const* ip = cmaes->getPtr(CMAES<fpt>::XBestEver);
      for(unsigned i = 0; i < opt->dimension(); i++)
      {
        optimum(i) = ip[i];
      }
      opt->setParameters(optimum);
      optimumValue = opt->error();
    }

    if(cmaes->get(CMAES<fpt>::FBestEver) < stop.minimalValue)
      break;
  }
}

Vt IPOPCMAES::getNext()
{
  OPENANN_CHECK(cmaes);
  OPENANN_CHECK(opt);
  OPENANN_CHECK_WITHIN(currentIndividual, 0, cmaes->get(CMAES<fpt>::Lambda)-1);

  if(currentIndividual == 0)
  {
    population = cmaes->samplePopulation();
  }

  Vt individual(opt->dimension());
  for(unsigned i = 0; i < opt->dimension(); i++)
  {
    individual(i) = population[currentIndividual][i];
  }
  return individual;
}

void IPOPCMAES::setError(fpt fitness)
{
  OPENANN_CHECK(cmaes);
  OPENANN_CHECK(opt);
  OPENANN_CHECK_WITHIN(currentIndividual, 0, cmaes->get(CMAES<fpt>::Lambda)-1);

  if(logger.isActive())
  {
    for(unsigned i = 0; i < opt->dimension(); i++)
    {
      logger << population[currentIndividual][i] << " ";
    }
    logger << fitness << "\n";
  }

  fitnessValues[currentIndividual++] = fitness;
  evaluationsAfterRestart++;

  if(currentIndividual == cmaes->get(CMAES<fpt>::Lambda))
  {
    cmaes->updateDistribution(fitnessValues);
    currentIndividual = 0;
    opt->finishedIteration();
  }
}

bool IPOPCMAES::step()
{
}

bool IPOPCMAES::terminated()
{
  return currentIndividual == 0 && (cmaes->testForTermination()
      || (maxFunEvalsActive && evaluationsAfterRestart >= stop.maximalFunctionEvaluations));
}

Vt IPOPCMAES::result()
{
  OPENANN_CHECK(cmaes);
  OPENANN_CHECK(opt);

  opt->setParameters(optimum);
  return optimum;
}

std::string IPOPCMAES::name()
{
  return "Covariance Matrix Adaption Evolution Strategies";
}

void IPOPCMAES::setSigma0(fpt sigma0)
{
  this->sigma0 = sigma0;
}

}
