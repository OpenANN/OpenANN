#include "MBSGDTestCase.h"
#include "optimization/Quadratic.h"
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/io/Logger.h>

void MBSGDTestCase::run()
{
  RUN(MBSGDTestCase, quadratic);
  RUN(MBSGDTestCase, restart);
}

void MBSGDTestCase::quadratic()
{
  OpenANN::MBSGD mbsgd;
  Quadratic<10> q;
  q.setParameters(Eigen::VectorXd::Ones(10));
  OpenANN::StoppingCriteria s;
  s.maximalIterations = 1000;
  s.minimalSearchSpaceStep = 1e-10;
  mbsgd.setOptimizable(q);
  mbsgd.setStopCriteria(s);
  mbsgd.optimize();
  Eigen::VectorXd optimum = mbsgd.result();
  ASSERT(q.error() < 0.001);
}

void MBSGDTestCase::restart()
{
  OpenANN::MBSGD mbsgd;
  Quadratic<10> q;
  q.setParameters(Eigen::VectorXd::Ones(10));
  OpenANN::StoppingCriteria s;
  s.maximalIterations = 1000;
  s.minimalSearchSpaceStep = 1e-10;
  mbsgd.setOptimizable(q);
  mbsgd.setStopCriteria(s);
  mbsgd.optimize();
  Eigen::VectorXd optimum = mbsgd.result();
  ASSERT(q.error() < 0.001);

  // Restart
  q.setParameters(Eigen::VectorXd::Ones(10));
  ASSERT(q.error() == 10.0);
  mbsgd.optimize();
  optimum = mbsgd.result();
  ASSERT(q.error() < 0.001);
}
