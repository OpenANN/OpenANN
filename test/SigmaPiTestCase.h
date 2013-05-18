#pragma once

#include "Test/TestCase.h"

class SigmaPiTestCase : public TestCase
{
  virtual void run();
  void sigmaPiNoConstraintGradient();
  void sigmaPiWithConstraintGradient();
};
