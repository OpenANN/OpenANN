#include <optimization/MBSGD.h>
#include <Optimizable.h>
#include <StopCriteria.h>
#include <AssertionMacros.h>
#include <Random.h>
#include <Test/Stopwatch.h>
#include <numeric>
#include <vector>
#include <list>

namespace OpenANN {

MBSGD::MBSGD()
  : debugLogger(Logger::CONSOLE), batchSize(10), alpha(0.001), eta(0.6),
    minGain(0.01), maxGain(100.0)
{
}

MBSGD::~MBSGD()
{
}

void MBSGD::setOptimizable(Optimizable& opt)
{
  this->opt = &opt;
}

void MBSGD::setStopCriteria(const StopCriteria& stop)
{
  this->stop = stop;
}

void MBSGD::optimize()
{
  OPENANN_CHECK(opt->providesInitialization());

  RandomNumberGenerator rng;
  const int P = opt->dimension();
  const unsigned N = opt->examples();
  const int batches = N / batchSize;
  Vt gradient(P);
  gradient.fill(0.0);
  Vt gains(P);
  gains.fill(1.0);
  Vt parameters = opt->currentParameters();
  Vt momentum(P);
  momentum.fill(0.0);
  std::vector<std::list<int> > batchAssignment(batches);
  int iteration = 1;

  Stopwatch sw;
  do {
    for(int n = 0; n < N; n++)
      batchAssignment[rng.generateIndex(batches)].push_back(n);
    for(int b = 0; b < batches; b++)
    {
      for(std::list<int>::const_iterator it = batchAssignment[b].begin();
          it != batchAssignment[b].end(); it++)
        gradient += opt->gradient(rng.generateIndex(*it));
      batchAssignment[b].clear();
      for(int p = 0; p < P; p++)
      {
        if(momentum(p)*gradient(p) >= (fpt) 0.0)
          gains(p) += 0.05;
        else
          gains(p) *= 0.95;
        gains(p) = std::min<fpt>(maxGain, std::max<fpt>(minGain, gains(p)));
        gradient(p) *= gains(p);
      }
      momentum = eta * momentum - alpha * gradient;
      parameters += momentum;
      opt->setParameters(parameters);
      gradient.fill(0.0);
      debugLogger << ".";
    }
    debugLogger << "\n";

    debugLogger << "iteration " << iteration++ << " finished in " << sw.stop(Stopwatch::MILLISECOND) << " ms\n\n";
    debugLogger << "gains = " << gains.transpose() << "\n\n";
    sw.start();
    opt->finishedIteration();
    if(debugLogger.isActive())
    {
      debugLogger << "test in " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";
      sw.start();
    }
  } while(stop.maximalIterations == StopCriteria::defaultValue.maximalFunctionEvaluations || iteration <= stop.maximalIterations);
}

Vt MBSGD::result()
{
  return optimum;
}

std::string MBSGD::name()
{
  return "Mini-Batch Stochastic Gradient Descent";
}

}
