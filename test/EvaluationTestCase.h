#ifndef OPENANN_TEST_EVALUATION_TEST_CASE_H_
#define OPENANN_TEST_EVALUATION_TEST_CASE_H_

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
  void crossValidation();
};

#endif // OPENANN_TEST_EVALUATION_TEST_CASE_H_
