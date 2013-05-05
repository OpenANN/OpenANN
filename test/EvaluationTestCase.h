#pragma once

#include <Test/TestCase.h>

class EvaluationTestCase : public TestCase
{
  virtual void run();
  virtual void setUp();
  void sse();
  void mse();
  void rmse();
  void ce();
  void accuracy();
  void confusionMatrix();
};
