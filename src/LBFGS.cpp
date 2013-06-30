#include <OpenANN/optimization/LBFGS.h>

namespace OpenANN
{

LBFGS::LBFGS(int m)
  : opt(0), iteration(-1), n(-1), m(m), error(0.0)
{
}

void LBFGS::setStopCriteria(const StoppingCriteria& sc)
{
  this->stop = stop;
}

void LBFGS::setOptimizable(Optimizable& optimizable)
{
  this->opt = &optimizable;
}

void LBFGS::optimize()
{
  OPENANN_CHECK(opt);
  while(step())
  {
    OPENANN_DEBUG << "Iteration #" << iteration << ", training error = "
                  << FloatingPointFormatter(error, 4);
  }
}

bool LBFGS::step()
{
  OPENANN_CHECK(opt);
  if(iteration < 0)
    initialize();
  OPENANN_CHECK(n > 0);

  try
  {
    while(alglib_impl::minlbfgsiteration(state.c_ptr(), &envState))
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

void LBFGS::initialize()
{
  n = opt->dimension();

  parameters.resize(n);
  gradient.resize(n);

  xIn.setcontent(n, opt->currentParameters().data());

  // Initialize optimizer
  alglib::minlbfgscreate(m, xIn, state);

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
  alglib::minlbfgssetcond(state, minimalSearchSpaceStep, minimalValueDifferences,
                          0.0, maximalIterations);

  // Initialize optimizer state
  alglib_impl::ae_state_init(&envState);

  iteration = 0;
}

void LBFGS::reset()
{
  alglib_impl::ae_state_clear(&envState);

  alglib::minlbfgsreport report;
  alglib::minlbfgsresults(state, xIn, report);
  optimum.resize(n, 1);
  for(unsigned i = 0; i < n; i++)
    optimum(i) = xIn[i];

  OPENANN_DEBUG << "CG terminated";
  OPENANN_DEBUG << report.iterationscount << " iterations";
  OPENANN_DEBUG << report.nfev << " function evaluations";
  OPENANN_DEBUG << "reason: ";
  switch(report.terminationtype)
  {
  case -2:
    OPENANN_DEBUG << "Rounding errors prevent further improvement.";
    break;
  case -1:
    OPENANN_DEBUG << "Incorrect parameters were specified.";
    break;
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
  default:
    OPENANN_DEBUG << "Unknown.";
  }

  iteration = -1;
}

Eigen::VectorXd LBFGS::result()
{
  OPENANN_CHECK(opt);
  opt->setParameters(optimum);
  return optimum;
}

std::string LBFGS::name()
{
  return "L-BFGS";
}

} // namespace OpenANN
