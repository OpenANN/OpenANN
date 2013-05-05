#pragma once

#include <Test/TestCase.h>

class CGTestCase : public TestCase
{
  virtual void run();
  void quadratic();
  void restart();
};
