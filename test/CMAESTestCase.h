#ifndef OPENANN_TEST_CMAES_TEST_CASE_H_
#define OPENANN_TEST_CMAES_TEST_CASE_H_

#include "Test/TestCase.h"

class CMAESTestCase : public TestCase
{
  virtual void run();
  void rosenbrock();
  void himmelblau();
  void ellinum();
};

#endif // OPENANN_TEST_CMAES_TEST_CASE_H_
