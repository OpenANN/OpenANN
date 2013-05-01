#define OPENANN_LOG_NAMESPACE "LMA"

#include <OpenANN/optimization/LMA.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/optimization/StoppingInterrupt.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/io/Logger.h>
#include <limits>

namespace OpenANN {

LMA::LMA()
    : opt(0), iteration(-1)
{
}

LMA::~LMA()
{
}

void LMA::setOptimizable(Optimizable& opt)
{
  this->opt = &opt;
}

void LMA::setStopCriteria(const StoppingCriteria& stop)
{
  this->stop = stop;
}

void LMA::optimize()
{
  OPENANN_CHECK(opt);

  OpenANN::StoppingInterrupt interrupt;

  while(step() && !interrupt.isSignaled())
  {
    OPENANN_DEBUG << "Iteration #" << iteration 
      << ", training error = " << FloatingPointFormatter(errorValues.sum(), 4);
  }
}

bool LMA::step()
{
  OPENANN_CHECK(opt);
  if(iteration < 0)
      initialize();

  try
  {
    while(alglib_impl::minlmiteration(state.c_ptr(), &envState))
    {
      if(state.needfi)
      {
        for(unsigned i = 0; i < n; i++)
          parameters(i) = state.x[i];
        opt->setParameters(parameters);
        for(unsigned i = 0; i < opt->examples(); i++)
          state.fi[i] = opt->error(i);
        if(iteration != state.c_ptr()->repiterationscount)
        {
          iteration = state.c_ptr()->repiterationscount;
          opt->finishedIteration();
          return true;
        }
        continue;
      }
      if(state.needfij)
      {
        for(unsigned i = 0; i < n; i++)
          parameters(i) = state.x[i];
        opt->setParameters(parameters);
        opt->VJ(errorValues, jacobian);
        for(unsigned ex = 0; ex < opt->examples(); ex++)
        {
          state.fi[ex] = errorValues(ex);
          OPENANN_CHECK_EQUALS(state.j.rows(), jacobian.rows());
          OPENANN_CHECK_EQUALS(state.j.cols(), jacobian.cols());
          for(unsigned d = 0; d < opt->dimension(); d++)
            state.j[ex][d] = jacobian(ex, d);
        }
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
      throw alglib::ap_error("ALGLIB: error in 'minlmoptimize' (some "
          "derivatives were not provided?)");
    }
    alglib_impl::ae_state_clear(&envState);
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

Eigen::VectorXd LMA::result()
{
  OPENANN_CHECK(opt);
  opt->setParameters(optimum);
  return optimum;
}

std::string LMA::name()
{
  std::stringstream stream;
  stream << "Levenberg-Marquardt Algorithm";
  return stream.str();
}

void LMA::initialize()
{
  n = opt->dimension();

  // temporary vectors to avoid allocations
  parameters.resize(n);
  errorValues.resize(opt->examples());
  jacobian.resize(opt->examples(), n);

  xIn.setcontent(n, opt->currentParameters().data());

  // Initialize optimizer
  alglib::minlmcreatevj(opt->examples(), xIn, state);

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
  alglib::minlmsetcond(state, minimalSearchSpaceStep, minimalValueDifferences,
      0.0, maximalIterations);

  // Initialize optimizer state
  alglib_impl::ae_state_init(&envState);
}

void LMA::reset()
{
  // Read out results
  alglib::minlmresults(state, xIn, report);

  // Set optimum
  optimum.resize(n);
  for(unsigned i = 0; i < n; i++)
    optimum(i) = xIn[i];
  opt->setParameters(optimum);

  // Log result
  OPENANN_DEBUG << "LMA terminated\n"
                << "Iterations= " << report.iterationscount << std::endl
                << "Function evaluations= " << report.nfunc << std::endl
                << "Jacobi evaluations= " << report.njac << std::endl
                << "Gradient evaluations= " << report.ngrad << std::endl
                << "Hessian evaluations= " << report.nhess << std::endl
                << "Cholesky decompositions= " << report.ncholesky << std::endl
                << "Value= " << opt->error() << std::endl
                << "Reason: ";
  switch(report.terminationtype)
  {
  case 1:
    OPENANN_DEBUG << "Relative function improvement is below threshold.";
    break;
  case 2:
    OPENANN_DEBUG << "Relative step is below threshold.";
    break;
  case 4:
    OPENANN_DEBUG << "Gradient is below threshold.";
    break;
  case 5:
    OPENANN_DEBUG << "MaxIts steps was taken";
    break;
  case 7:
    OPENANN_DEBUG << "Stopping conditions are too stringent, "
                << "further improvement is impossible.";
    break;
  default:
    OPENANN_DEBUG << "Unknown.";
  }

  iteration = -1;
}

}
