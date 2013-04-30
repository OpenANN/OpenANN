#include <OpenANN/optimization/CG.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/Random.h>
#include <limits>
#include "optimization.h"
#include "Test/Stopwatch.h"

namespace OpenANN {

CG::CG() : debugLogger(Logger::CONSOLE), opt(0)
{
}

CG::~CG()
{
}

void CG::setOptimizable(Optimizable& opt)
{
  this->opt = &opt;
}

void CG::setStopCriteria(const StoppingCriteria& stop)
{
  this->stop = stop;
}

void CG::optimize()
{
  OPENANN_CHECK(opt);
  const unsigned n = opt->dimension();
  if(!opt->providesInitialization())
  {
    Eigen::VectorXd init(n);
    RandomNumberGenerator rng;
    for(unsigned i = 0; i < n; i++)
      init(i) = rng.sampleNormalDistribution<double>();
    opt->setParameters(init);
  }

  alglib::mincgstate state;
  alglib::mincgreport report;
  alglib::real_1d_array xIn;
  double* lengthIndicator = new double[n];
  std::memset(lengthIndicator, 0, n*sizeof(double));
  xIn.setcontent(n, lengthIndicator);
  delete[] lengthIndicator;
  for(unsigned i = 0; i < n; i++)
    xIn[i] = opt->currentParameters()(i);

  alglib::mincgcreate(xIn, state);
  alglib::mincgsetcond(state,
      stop.minimalSearchSpaceStep != StoppingCriteria::defaultValue.minimalSearchSpaceStep ?
          stop.minimalSearchSpaceStep : 0.0,
      stop.minimalValueDifferences != StoppingCriteria::defaultValue.minimalValueDifferences ?
          stop.minimalValueDifferences : 0.0,
      0.0,
      stop.maximalIterations != StoppingCriteria::defaultValue.maximalIterations ?
          stop.maximalIterations : 0);

  // temporary vectors to avoid allocations
  Eigen::VectorXd parameters(n);
  Eigen::VectorXd gradient(n);

  int iteration = -1;
  alglib_impl::ae_state _alglib_env_state;
  alglib_impl::ae_state_init(&_alglib_env_state);
  try
  {
    Stopwatch optimizerStopWatch;
    while(alglib_impl::mincgiteration(state.c_ptr(), &_alglib_env_state))
    {
      if(state.needfg)
      {
        debugLogger << "computed optimization step in " << optimizerStopWatch.stop(Stopwatch::MILLISECOND) << " ms\n";
        for(unsigned i = 0; i < n; i++)
          parameters(i) = state.x[i];
        opt->setParameters(parameters);
        optimizerStopWatch.start();
        state.f = opt->error();
        debugLogger << "computed error in " << optimizerStopWatch.stop(Stopwatch::MILLISECOND) << " ms\n";
        optimizerStopWatch.start();
        gradient = opt->gradient();
        debugLogger << "computed gradient in " << optimizerStopWatch.stop(Stopwatch::MILLISECOND) << " ms\n";
        for(unsigned i = 0; i < n; i++)
          state.g[i] = (double) gradient(i, 0);
        if(iteration != state.c_ptr()->repiterationscount)
        {
          iteration = state.c_ptr()->repiterationscount;
          opt->finishedIteration();
        }
        optimizerStopWatch.start();
        continue;
      }
      if(state.xupdated)
      {
        continue;
      }
      throw alglib::ap_error("ALGLIB: error in 'mincgoptimize' (some derivatives were not provided?)");
    }
    alglib_impl::ae_state_clear(&_alglib_env_state);
  }
  catch(alglib_impl::ae_error_type)
  {
    throw alglib::ap_error(_alglib_env_state.error_msg);
  }
  catch(...)
  {
    throw;
  }

  alglib::mincgresults(state, xIn, report);
  optimum.resize(n, 1);
  for(unsigned i = 0; i < n; i++)
    optimum(i, 0) = xIn[i];

  if(debugLogger.isActive())
  {
    debugLogger << "CG terminated\n"
                << "iterations= " << report.iterationscount << "\n"
                << "function evaluations= " << report.nfev << "\n"
                //<< "optimum= " << optimum.transpose() << "\n"
                << "reason: ";
    switch(report.terminationtype)
    {
    case 1:
      debugLogger << "Relative function improvement is no more than EpsF.\n";
      break;
    case 2:
      debugLogger << "Relative step is no more than EpsX.\n";
      break;
    case 4:
      debugLogger << "Gradient norm is no more than EpsG.\n";
      break;
    case 5:
      debugLogger << "MaxIts steps was taken.\n";
      break;
    case 7:
      debugLogger << "Stopping conditions are too stringent, further"
                  << "further improvement is impossible, we return best "
                  << "X found so far.\n";
      break;
    case 8:
      debugLogger << "Terminated by user.\n";
      break;
    default:
      debugLogger << "Unknown.\n";
    }
  }
}

Eigen::VectorXd CG::result()
{
  OPENANN_CHECK(opt);
  opt->setParameters(optimum);
  return optimum;
}

std::string CG::name()
{
  return "Conjugate Gradient";
}

}
