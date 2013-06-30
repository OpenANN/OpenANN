#ifndef OPENANN_TEST_SPARSE_AUTO_ENCODER_TEST_CASE_H_
#define OPENANN_TEST_SPARSE_AUTO_ENCODER_TEST_CASE_H_

#include <Test/TestCase.h>

class SparseAutoEncoderTestCase : public TestCase
{
  virtual void run();
  void gradient();
};

#endif // OPENANN_TEST_SPARSE_AUTO_ENCODER_TEST_CASE_H_
