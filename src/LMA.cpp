#ifdef USE_GPL_LICENSE

#include <optimization/LMA.h>
#include <Optimizable.h>
#include <AssertionMacros.h>
#include <Random.h>
#include <limits>

namespace OpenANN {

LMA::LMA(bool approximateHessian)
    : debugLogger(Logger::NONE), opt(0),
      approximateHessian(approximateHessian), iteration(-1)
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
  optimizerStopWatch.start();
}

void LMA::setStopCriteria(const StoppingCriteria& stop)
{
  this->stop = stop;
}

void LMA::optimize()
{
  OPENANN_CHECK(opt);
  while(step());
}

bool LMA::step()
{
  try
  {
    while(alglib_impl::minlmiteration(state.c_ptr(), &_alglib_env_state))
    {
      debugLogger << "computed optimization step in "
          << optimizerStopWatch.stop(Stopwatch::MILLISECOND) << " ms\n";
      if(state.needfi)
      {
        for(unsigned i = 0; i < n; i++)
          parameters(i) = (fpt) state.x[i];
        opt->setParameters(parameters);
        optimizerStopWatch.start();
        for(unsigned i = 0; i < opt->examples(); i++)
          state.fi[i] = (double) opt->error(i);
        debugLogger << "computed errors in " << optimizerStopWatch.stop(Stopwatch::MILLISECOND) << " ms\n";
        if(iteration != state.c_ptr()->repiterationscount)
        {
          optimizerStopWatch.start();
          iteration = state.c_ptr()->repiterationscount;
          opt->finishedIteration();
          debugLogger << "finished iteration in " << optimizerStopWatch.stop(Stopwatch::MILLISECOND) << " ms\n";
        }
        optimizerStopWatch.start();
        return true;
      }
      if(state.needfij)
      {
        for(unsigned i = 0; i < n; i++)
          parameters(i) = (fpt) state.x[i];
        opt->setParameters(parameters);
        optimizerStopWatch.start();
        opt->VJ(errorValues, jacobian);
        debugLogger << "computed errors and jacobian in " << optimizerStopWatch.stop(Stopwatch::MILLISECOND) << " ms\n";
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
          optimizerStopWatch.start();
          iteration = state.c_ptr()->repiterationscount;
          opt->finishedIteration();
          debugLogger << "finished iteration in " << optimizerStopWatch.stop(Stopwatch::MILLISECOND) << " ms\n";
        }
        optimizerStopWatch.start();
        return true;
      }
      if(!approximateHessian)
      {
        if(state.needf)
        {
          for(unsigned i = 0; i < n; i++)
            parameters(i) = (fpt) state.x[i];
          opt->setParameters(parameters);
          state.f = (double) opt->error();
          if(iteration != state.c_ptr()->repiterationscount)
          {
            iteration = state.c_ptr()->repiterationscount;
            opt->finishedIteration();
          }
        }
        if(state.needfg)
        {
          for(unsigned i = 0; i < n; i++)
            parameters(i) = (fpt) state.x[i];
          opt->setParameters(parameters);
          state.f = opt->error();
          gradient = opt->gradient();
          for(int i = 0; i < gradient.rows(); i++)
            state.g[i] = (double) gradient(i);
          if(iteration != state.c_ptr()->repiterationscount)
          {
            iteration = state.c_ptr()->repiterationscount;
            opt->finishedIteration();
          }
        }
        if(state.needfgh)
        {
          for(unsigned i = 0; i < n; i++)
            parameters(i) = (fpt) state.x[i];
          opt->setParameters(parameters);
          state.f = (double) opt->error();
          gradient = opt->gradient();
          for(int i = 0; i < gradient.rows(); i++)
            state.g[i] = (double) gradient(i);
          Mt hess = opt->hessian();
          for(int i = 0; i < hess.rows(); i++)
            for(int j = 0; j < hess.cols(); j++)
              state.h[i][j] = hess(i, j);
          if(iteration != state.c_ptr()->repiterationscount)
          {
            iteration = state.c_ptr()->repiterationscount;
            opt->finishedIteration();
          }
          return true;
        }
      }
      if(state.xupdated)
        return true;
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
  stream << "Levenberg-Marquardt Algorithm (" << (approximateHessian ? "appr.)" : "exact)");
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
  if(approximateHessian)
    alglib::minlmcreatevj(opt->examples(), xIn, state);
  else
    alglib::minlmcreatefgh(xIn, state);

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

  alglib_impl::ae_state_init(&_alglib_env_state);
}

void LMA::cleanUp()
{
  alglib::minlmresults(state, xIn, report);
  optimum.resize(n);
  for(unsigned i = 0; i < n; i++)
    optimum(i) = xIn[i];
  opt->setParameters(optimum);

  if(debugLogger.isActive())
  {
    debugLogger << "LMA terminated\n"
                << "iterations= " << report.iterationscount << "\n"
                << "function evaluations= " << report.nfunc << "\n"
                << "jacobi evaluations= " << report.njac << "\n"
                << "gradient evaluations= " << report.ngrad << "\n"
                << "hessian evaluations= " << report.nhess << "\n"
                << "Cholesky decompositions= " << report.ncholesky << "\n"
                << "value= " << opt->error() << "\n"
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
      debugLogger << "Gradient is no more than EpsG.\n";
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
