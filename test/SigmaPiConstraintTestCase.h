#pragma once

#include "Test/TestCase.h"
#include <Eigen/Dense>

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

