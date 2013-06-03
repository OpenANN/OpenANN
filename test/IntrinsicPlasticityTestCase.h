#ifndef OPENANN_TEST_INTRINSIC_PLASTICITY_TEST_CASE_H_
#define OPENANN_TEST_INTRINSIC_PLASTICITY_TEST_CASE_H_

#include <Test/TestCase.h>

class IntrinsicPlasticityTestCase : public TestCase
{
  virtual void run();
  virtual void setUp();
  void learn();
  void backprop();
};

#endif // OPENANN_TEST_INTRINSIC_PLASTICITY_TEST_CASE_H_
