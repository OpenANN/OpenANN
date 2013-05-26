#pragma once

#include "Test/TestCase.h"

class LocalResponseNormalizationTestCase : public TestCase
{
  virtual void run();
  void localResponseNormalizationInputGradient();
};
