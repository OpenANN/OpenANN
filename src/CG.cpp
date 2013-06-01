#define OPENANN_LOG_NAMESPACE "CG"

#include <OpenANN/optimization/CG.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/io/Logger.h>
#include <limits>

namespace OpenANN
{

CG::CG()
  : opt(0), iteration(-1), error(0.0)
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

bool CG::step()
{
  OPENANN_CHECK(opt);
  if(iteration < 0)
    initialize();

  try
  {
    while(alglib_impl::mincgiteration(state.c_ptr(), &envState))
    {
      if(state.needf)
      {
        for(unsigned i = 0; i < n; i++)
          parameters(i) = state.x[i];
        opt->setParameters(parameters);
        error = opt->error();
        state.f = error;
        if(iteration != state.c_ptr()->repiterationscount)
        {
          iteration = state.c_ptr()->repiterationscount;
          opt->finishedIteration();
          return true;
        }
        continue;
      }
      if(state.needfg)
      {
        for(unsigned i = 0; i < n; i++)
          parameters(i) = state.x[i];
        opt->setParameters(parameters);
        opt->errorGradient(error, gradient);
        state.f = error;
        for(unsigned i = 0; i < n; i++)
          state.g[i] = (double) gradient(i);
        if(iteration != state.c_ptr()->repiterationscount)
        {
          iteration = state.c_ptr()->repiterationscount;
          opt->finishedIteration();
          return true;
        }
        continue;
      }
      if(state.xupdated)
        continue;
      throw alglib::ap_error("ALGLIB: error in 'mincgoptimize'"
                             " (some derivatives were not provided?)");
    }
  }
  catch(alglib_impl::ae_error_type)
  {
    throw OpenANNException(envState.error_msg);
  }
  catch(...)
  {
    throw;
  }

  reset();

  return false;
}

void CG::optimize()
{
  OPENANN_CHECK(opt);
  while(step())
  {
    OPENANN_DEBUG << "Iteration #" << iteration << ", training error = "
                  << FloatingPointFormatter(error, 4);
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

void CG::initialize()
{
  n = opt->dimension();

  parameters.resize(n);
  gradient.resize(n);

  xIn.setcontent(n, opt->currentParameters().data());

  // Initialize optimizer
  alglib::mincgcreate(xIn, state);

  // Set convergence criteria
  double minimalSearchSpaceStep = stop.minimalSearchSpaceStep !=
                                  StoppingCriteria::defaultValue.minimalSearchSpaceStep ?
                                  stop.minimalSearchSpaceStep : 0.0;
  double minimalValueDifferences = stop.minimalValueDifferences !=
                                   StoppingCriteria::defaultValue.minimalValueDifferences ?
                                   stop.minimalValueDifferences : 0.0;
  int maximalIterations = stop.maximalIterations !=
                          StoppingCriteria::defaultValue.maximalIterations ?
                          stop.maximalIterations : 0;
  alglib::mincgsetcond(state, minimalSearchSpaceStep, minimalValueDifferences,
                       0.0, maximalIterations);

  // Initialize optimizer state
  alglib_impl::ae_state_init(&envState);

  iteration = 0;
}

void CG::reset()
{
  alglib_impl::ae_state_clear(&envState);

  alglib::mincgresults(state, xIn, report);
  optimum.resize(n, 1);
  for(unsigned i = 0; i < n; i++)
    optimum(i) = xIn[i];

  OPENANN_DEBUG << "CG terminated";
  OPENANN_DEBUG << report.iterationscount << " iterations";
  OPENANN_DEBUG << report.nfev << " function evaluations";
  OPENANN_DEBUG << "reason: ";
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
                  << " improvement is impossible, we return the best "
                  << "X found so far.";
    break;
  case 8:
    OPENANN_DEBUG << "Terminated by user.";
    break;
  default:
    OPENANN_DEBUG << "Unknown.";
  }

  iteration = -1;
}

} // namespace OpenANN
