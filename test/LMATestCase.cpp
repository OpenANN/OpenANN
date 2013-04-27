#include "LMATestCase.h"
#include "optimization/Quadratic.h"
#include <OpenANN/optimization/LMA.h>
#include <OpenANN/io/Logger.h>

void LMATestCase::run()
{
  RUN(LMATestCase, quadratic);
  RUN(LMATestCase, restart);
}

void LMATestCase::quadratic()
{
  OpenANN::LMA lma;
  Quadratic<10> q;
  q.setParameters(Eigen::VectorXd::Ones(10));
  OpenANN::StoppingCriteria s;
  s.maximalIterations = 1000;
  s.minimalSearchSpaceStep = 1e-10;
  lma.setOptimizable(q);
  lma.setStopCriteria(s);
  lma.optimize();
  Eigen::VectorXd optimum = lma.result();
  ASSERT(q.error() < 0.001);
}

void LMATestCase::restart()
{
  OpenANN::LMA lma;
  Quadratic<10> q;
  q.setParameters(Eigen::VectorXd::Ones(10));
  OpenANN::StoppingCriteria s;
  s.maximalIterations = 1000;
  s.minimalSearchSpaceStep = 1e-10;
  lma.setOptimizable(q);
  lma.setStopCriteria(s);
  lma.optimize();
  Eigen::VectorXd optimum = lma.result();
  ASSERT(q.error() < 0.001);

  // Restart
  q.setParameters(Eigen::VectorXd::Ones(10));
  ASSERT(q.error() == 10.0);
  lma.optimize();
  optimum = lma.result();
  ASSERT(q.error() < 0.001);
}
