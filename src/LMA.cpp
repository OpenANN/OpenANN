#ifdef USE_GPL_LICENSE

#include <optimization/LMA.h>
#include <Optimizable.h>
#include <AssertionMacros.h>
#include <Random.h>
#include <limits>
#include <optimization.h>
#include <Test/Stopwatch.h>

namespace OpenANN {

LMA::LMA(bool approximateHessian)
    : debugLogger(Logger::CONSOLE), opt(0),
      approximateHessian(approximateHessian)
{
}

LMA::~LMA()
{
}

void LMA::setOptimizable(Optimizable& opt)
{
  this->opt = &opt;
}

void LMA::setStopCriteria(const StopCriteria& stop)
{
  this->stop = stop;
}

void LMA::optimize()
{
  OPENANN_CHECK(opt);

  const unsigned n = opt->dimension();
  if(opt->providesInitialization())
  {
    opt->initialize();
  }
  else
  {
    RandomNumberGenerator rng;
    Vt x(n);
    for(unsigned i = 0; i < n; i++)
      x(i) = rng.sampleNormalDistribution<fpt>();
  }

  double* xArray = new double[n];
  Vt x = opt->currentParameters();
  for(unsigned i = 0; i < n; i++)
    xArray[i] = x(i);

  alglib::minlmstate state;
  alglib::minlmreport report;
  alglib::real_1d_array xIn;
  xIn.setcontent(n, xArray);
  delete[] xArray;

  if(approximateHessian)
  {
    const unsigned examples = opt->examples();
    alglib::minlmcreatevj(examples, xIn, state);
  }
  else
  {
    alglib::minlmcreatefgh(xIn, state);
  }

  alglib::minlmsetcond(state,
      stop.minimalSearchSpaceStep != StopCriteria::defaultValue.minimalSearchSpaceStep ?
          stop.minimalSearchSpaceStep : 0.0,
      stop.minimalValueDifferences != StopCriteria::defaultValue.minimalValueDifferences ?
          stop.minimalValueDifferences : 0.0,
      0.0,
      stop.maximalIterations != StopCriteria::defaultValue.maximalIterations ?
          stop.maximalIterations : 0);

  // temporary vectors to avoid allocations
  Vt parameters(n);
  Vt gradient(n);
  Vt errorValues(opt->examples());
  Mt jacobian(opt->examples(), n);

  int iteration = -1;
  alglib_impl::ae_state _alglib_env_state;
  alglib_impl::ae_state_init(&_alglib_env_state);
  try
  {
    Stopwatch optimizerStopWatch;
    while(alglib_impl::minlmiteration(state.c_ptr(), &_alglib_env_state))
    {
      debugLogger << "computed optimization step in " << optimizerStopWatch.stop(Stopwatch::MILLISECOND) << " ms\n";
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
        continue;
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
        continue;
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
          continue;
        }
      }
      if(state.xupdated)
        continue;
      throw alglib::ap_error("ALGLIB: error in 'minlmoptimize' (some derivatives were not provided?)");
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
  alglib::minlmresults(state, xIn, report);
  optimum.resize(n);
  for(unsigned i = 0; i < n; i++)
    optimum(i) = xIn[i];

  if(debugLogger.isActive())
  {
    opt->setParameters(optimum);
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

}

#endif // USE_GPL_LICENSE
