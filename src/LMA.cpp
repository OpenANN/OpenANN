#ifdef USE_GPL_LICENSE

#include <optimization/LMA.h>
#include <Optimizable.h>
#include <AssertionMacros.h>
#include <Random.h>
#include <limits>

namespace OpenANN {

LMA::LMA()
    : debugLogger(Logger::CONSOLE), opt(0), iteration(-1)
{
}

LMA::~LMA()
{
}

void LMA::setOptimizable(Optimizable& opt)
{
  this->opt = &opt;
  initialize();
  allocate();
  initALGLIB();
}

void LMA::setStopCriteria(const StoppingCriteria& stop)
{
  this->stop = stop;
}

void LMA::optimize()
{
  OPENANN_CHECK(opt);
  while(step())
  {
    if(debugLogger.isActive())
    {
      debugLogger << "Iteration " << iteration << " finished.\n";
      debugLogger << "Error = " << errorValues.sum() << ".\n";
    }
  }
}

bool LMA::step()
{
  try
  {
    while(alglib_impl::minlmiteration(state.c_ptr(), &_alglib_env_state))
    {
      if(state.needfi)
      {
        for(unsigned i = 0; i < n; i++)
          parameters(i) = (fpt) state.x[i];
        opt->setParameters(parameters);
        for(unsigned i = 0; i < opt->examples(); i++)
          state.fi[i] = (double) opt->error(i);
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
          parameters(i) = (fpt) state.x[i];
        opt->setParameters(parameters);
        opt->VJ(errorValues, jacobian);
        for(unsigned ex = 0; ex < opt->examples(); ex++)
        {
          state.fi[ex] = (double) errorValues(ex);
          OPENANN_CHECK_EQUALS(state.j.rows(), jacobian.rows());
          OPENANN_CHECK_EQUALS(state.j.cols(), jacobian.cols());
          for(unsigned d = 0; d < opt->dimension(); d++)
            state.j[ex][d] = (double) jacobian(ex, d);
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

  cleanUp();
  return false;
}

Vt LMA::result()
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
  if(opt->providesInitialization())
    opt->initialize();
  else
  {
    RandomNumberGenerator rng;
    Vt x(n);
    for(unsigned i = 0; i < n; i++)
      x(i) = rng.sampleNormalDistribution<fpt>();
    opt->setParameters(x);
  }
}

void LMA::allocate()
{
  // temporary vectors to avoid allocations
  parameters.resize(n);
  gradient.resize(n);
  errorValues.resize(opt->examples());
  jacobian.resize(opt->examples(), n);

  double* xArray = new double[n];
  Vt x = opt->currentParameters();
  for(unsigned i = 0; i < n; i++)
    xArray[i] = x(i);
  xIn.setcontent(n, xArray);
  delete[] xArray;
}

void LMA::initALGLIB()
{
  // Initialize optimizer
  alglib::minlmcreatevj(opt->examples(), xIn, state);

  // Set convergence criteria
  fpt minimalSearchSpaceStep = stop.minimalSearchSpaceStep !=
      StoppingCriteria::defaultValue.minimalSearchSpaceStep ?
      stop.minimalSearchSpaceStep : 0.0;
  fpt minimalValueDifferences = stop.minimalValueDifferences !=
      StoppingCriteria::defaultValue.minimalValueDifferences ?
      stop.minimalValueDifferences : 0.0;
  int maximalIterations = stop.maximalIterations !=
      StoppingCriteria::defaultValue.maximalIterations ?
      stop.maximalIterations : 0;
  alglib::minlmsetcond(state, minimalSearchSpaceStep, minimalValueDifferences,
      0.0, maximalIterations);

  // Initialize optimizer state
  alglib_impl::ae_state_init(&_alglib_env_state);
}

void LMA::cleanUp()
{
  // Read out results
  alglib::minlmresults(state, xIn, report);

  // Set optimum
  optimum.resize(n);
  for(unsigned i = 0; i < n; i++)
    optimum(i) = xIn[i];
  opt->setParameters(optimum);

  // Log result
  if(debugLogger.isActive())
  {
    debugLogger << "LMA terminated\n"
                << "Iterations= " << report.iterationscount << "\n"
                << "Function evaluations= " << report.nfunc << "\n"
                << "Jacobi evaluations= " << report.njac << "\n"
                << "Gradient evaluations= " << report.ngrad << "\n"
                << "Hessian evaluations= " << report.nhess << "\n"
                << "Cholesky decompositions= " << report.ncholesky << "\n"
                << "Value= " << opt->error() << "\n"
                << "Reason: ";
    switch(report.terminationtype)
    {
    case 1:
      debugLogger << "Relative function improvement is below threshold.\n";
      break;
    case 2:
      debugLogger << "Relative step is below threshold.\n";
      break;
    case 4:
      debugLogger << "Gradient is below threshold.\n";
      break;
    case 5:
      debugLogger << "MaxIts steps was taken\n";
      break;
    case 7:
      debugLogger << "Stopping conditions are too stringent, "
                  << "further improvement is impossible,\n";
      break;
    default:
      debugLogger << "Unknown.\n";
    }
  }

  iteration = -1;
}

}

#endif // USE_GPL_LICENSE
