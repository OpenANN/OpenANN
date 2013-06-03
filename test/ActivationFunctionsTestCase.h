#ifndef OPENANN_TEST_ACTIVATION_FUNCTIONS_TEST_CASE_H_
#define OPENANN_TEST_ACTIVATION_FUNCTIONS_TEST_CASE_H_

#include <Test/TestCase.h>

class ActivationFunctionsTestCase : public TestCase
{
  virtual void run();
  void softmax();
  void logistic();
  void normaltanh();
  void linear();
  void rectifier();
};

#endif // OPENANN_TEST_ACTIVATION_FUNCTIONS_TEST_CASE_H_
