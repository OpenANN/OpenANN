#pragma once

#include <Test/TestCase.h>

class ActivationFunctionsTestCase : public TestCase
{
  virtual void run();
  void softmax();
  void logistic();
  void normaltanh();
  void linear();
};
