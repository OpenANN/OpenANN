#pragma once

#include "Test/TestCase.h"

class CMAESTestCase : public TestCase
{
  virtual void run();
  void rosenbrock();
  void himmelblau();
  void ellinum();
};
