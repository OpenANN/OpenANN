#ifndef OPENANN_OPTIMIZATION_LBFGS_H_
#define OPENANN_OPTIMIZATION_LBFGS_H_

#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/io/Logger.h>
#include <Eigen/Dense>
#include <optimization.h>

namespace OpenANN
{

/**
 * @class LBFGS
 *
 * Limited storage Broyden-Fletcher-Goldfarb-Shanno.
 *
 * L-BFGS is a quasi-Newton optimization algorithm that uses a low-rank
 * approximation of the Hessian (second derivative).
 */
class LBFGS : public Optimizer
{
  StoppingCriteria stop;
  Optimizable* opt; // do not delete
  Eigen::VectorXd optimum;
  int iteration, n, m;
  Eigen::VectorXd parameters, gradient;
  double error;
  alglib_impl::ae_state envState;
  alglib::minlbfgsstate state;
  alglib::real_1d_array xIn;
public:
  /**
   * Create L-BFGS optimizer.
   * @param m Number of corrections of the Hessian approximation update in
   *          BFGS scheme. Small values cause worse convergence, bigger values
   *          will not cause a considerably better convergence, but will
   *          decrease the performance.
   */
  LBFGS(int m = 10)
    : opt(0), iteration(-1), n(-1), m(m), error(0.0)
  {
    // TODO check parameter
  }

  virtual ~LBFGS()
  {
  }

  virtual void setStopCriteria(const StoppingCriteria& sc)
  {
    this->stop = stop;
  }

  virtual void setOptimizable(Optimizable& optimizable)
  {
    this->opt = &optimizable;
  }

  virtual void optimize()
  {
    OPENANN_CHECK(opt);
    while(step())
    {
      OPENANN_DEBUG << "Iteration #" << iteration << ", training error = "
                    << FloatingPointFormatter(error, 4);
    }
  }

  virtual bool step()
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

  void initialize()
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

  void reset()
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

  virtual Eigen::VectorXd result()
  {
    OPENANN_CHECK(opt);
    opt->setParameters(optimum);
    return optimum;
  }

  virtual std::string name()
  {
    return "L-BFGS";
  }
};

} // namespace OpenANN

#endif // OPENANN_OPTIMIZATION_LBFGS_H_
