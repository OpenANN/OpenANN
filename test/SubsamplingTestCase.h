#pragma once

#include <Test/TestCase.h>

class SubsamplingTestCase : public TestCase
{
  virtual void run();
  void subsampling();
  void subsamplingGradient();
  void subsamplingInputGradient();
};
