#include "CGTestCase.h"
#include "optimization/Quadratic.h"
#include <OpenANN/optimization/CG.h>

void CGTestCase::run()
{
  RUN(CGTestCase, quadratic);
  RUN(CGTestCase, restart);
}

void CGTestCase::quadratic()
{
  OpenANN::CG cg;
  Quadratic<10> q;
  q.setParameters(Eigen::VectorXd::Ones(10));
  OpenANN::StoppingCriteria s;
  s.maximalIterations = 1000;
  s.minimalSearchSpaceStep = 1e-10;
  cg.setOptimizable(q);
  cg.setStopCriteria(s);
  cg.optimize();
  Eigen::VectorXd optimum = cg.result();
  ASSERT(q.error() < 0.001);
}

void CGTestCase::restart()
{
  OpenANN::CG cg;
  Quadratic<10> q;
  q.setParameters(Eigen::VectorXd::Ones(10));
  OpenANN::StoppingCriteria s;
  s.maximalIterations = 1000;
  s.minimalSearchSpaceStep = 1e-10;
  cg.setOptimizable(q);
  cg.setStopCriteria(s);
  cg.optimize();
  Eigen::VectorXd optimum = cg.result();
  ASSERT(q.error() < 0.001);

  // Restart
  q.setParameters(Eigen::VectorXd::Ones(10));
  ASSERT(q.error() == 10.0);
  cg.optimize();
  optimum = cg.result();
  ASSERT(q.error() < 0.001);
}
