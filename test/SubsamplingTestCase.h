#ifndef OPENANN_TEST_SUBSAMPLING_TEST_CASE_H_
#define OPENANN_TEST_SUBSAMPLING_TEST_CASE_H_

#include <Test/TestCase.h>

class SubsamplingTestCase : public TestCase
{
  virtual void run();
  void subsampling();
  void subsamplingGradient();
  void subsamplingInputGradient();
};

#endif // OPENANN_TEST_SUBSAMPLING_TEST_CASE_H_
