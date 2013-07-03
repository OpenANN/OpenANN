#include "LBFGSTestCase.h"
#include "optimization/Quadratic.h"
#include <OpenANN/optimization/LBFGS.h>

void LBFGSTestcase::run()
{
  RUN(LBFGSTestcase, quadratic);
  RUN(LBFGSTestcase, restart);
}

void LBFGSTestcase::quadratic()
{
  OpenANN::LBFGS lbfgs;
  Quadratic<10> q;
  q.setParameters(Eigen::VectorXd::Ones(10));
  OpenANN::StoppingCriteria s;
  s.maximalIterations = 1000;
  s.minimalSearchSpaceStep = 1e-10;
  lbfgs.setOptimizable(q);
  lbfgs.setStopCriteria(s);
  lbfgs.optimize();
  Eigen::VectorXd optimum = lbfgs.result();
  ASSERT(q.error() < 0.001);
}

void LBFGSTestcase::restart()
{
  OpenANN::LBFGS lbfgs;
  Quadratic<10> q;
  q.setParameters(Eigen::VectorXd::Ones(10));
  OpenANN::StoppingCriteria s;
  s.maximalIterations = 1000;
  s.minimalSearchSpaceStep = 1e-10;
  lbfgs.setOptimizable(q);
  lbfgs.setStopCriteria(s);
  lbfgs.optimize();
  Eigen::VectorXd optimum = lbfgs.result();
  ASSERT(q.error() < 0.001);

  // Restart
  q.setParameters(Eigen::VectorXd::Ones(10));
  ASSERT(q.error() == 10.0);
  lbfgs.optimize();
  optimum = lbfgs.result();
  ASSERT(q.error() < 0.001);
}
