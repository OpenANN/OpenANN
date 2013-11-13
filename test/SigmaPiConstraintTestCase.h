#ifndef OPENANN_TEST_SIGMA_PI_CONSTRAINT_TEST_CASE_H_
#define OPENANN_TEST_SIGMA_PI_CONSTRAINT_TEST_CASE_H_

#include "Test/TestCase.h"
#include <Eigen/Core>

class SigmaPiConstraintTestCase : public TestCase
{
public:
  SigmaPiConstraintTestCase();

  virtual void run();

  void distance();
  void slope();
  void triangle();

  Eigen::VectorXd T1;
  Eigen::VectorXd T2;
  Eigen::VectorXd T3;
};

#endif // OPENANN_TEST_SIGMA_PI_CONSTRAINT_TEST_CASE_H_
