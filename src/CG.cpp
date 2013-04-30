#include <OpenANN/optimization/CG.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/io/Logger.h>
#include <limits>
#include "optimization.h"

namespace OpenANN {

CG::CG()
  : opt(0), iteration(-1)
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
  if(iteration < 0)
    initialize();

  alglib::mincgstate state;
  alglib::mincgreport report;
  alglib::real_1d_array xIn;
  double lengthIndicator[n];
  std::memset(lengthIndicator, 0, n*sizeof(double));
  xIn.setcontent(n, lengthIndicator);
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

  alglib_impl::ae_state _alglib_env_state;
  alglib_impl::ae_state_init(&_alglib_env_state);
  try
  {
    while(alglib_impl::mincgiteration(state.c_ptr(), &_alglib_env_state))
    {
      if(state.needfg)
      {
        for(unsigned i = 0; i < n; i++)
          parameters(i) = state.x[i];
        opt->setParameters(parameters);
        state.f = opt->error();
        gradient = opt->gradient();
        for(unsigned i = 0; i < n; i++)
          state.g[i] = (double) gradient(i, 0);
        if(iteration != state.c_ptr()->repiterationscount)
        {
          iteration = state.c_ptr()->repiterationscount;
          opt->finishedIteration();
        }
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

  if(OpenANN::Log::DEBUG  <= OpenANN::Log::getLevel())
  {
    OPENANN_DEBUG << "CG terminated" << std::endl
                << "iterations= " << report.iterationscount << std::endl
                << "function evaluations= " << report.nfev << std::endl
                << "reason: ";
    switch(report.terminationtype)
    {
    case 1:
      OPENANN_DEBUG << "Relative function improvement is no more than EpsF.";
      break;
    case 2:
      OPENANN_DEBUG << "Relative step is no more than EpsX.";
      break;
    case 4:
      OPENANN_DEBUG << "Gradient norm is no more than EpsG.";
      break;
    case 5:
      OPENANN_DEBUG << "MaxIts steps was taken.";
      break;
    case 7:
      OPENANN_DEBUG << "Stopping conditions are too stringent, further"
                  << "further improvement is impossible, we return best "
                  << "X found so far.";
      break;
    case 8:
      OPENANN_DEBUG << "Terminated by user.";
      break;
    default:
      OPENANN_DEBUG << "Unknown.";
    }
  }

  iteration = -1;
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

void CG::initialize()
{
  iteration = 0;
  n = opt->dimension();
}

}
