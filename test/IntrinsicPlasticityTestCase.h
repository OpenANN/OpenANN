#pragma once

#include <Test/TestCase.h>

class IntrinsicPlasticityTestCase : public TestCase
{
  virtual void run();
  virtual void setUp();
  void learn();
};
