#ifndef OPENANN_TEST_COMPRESSED_TEST_CASE_H_
#define OPENANN_TEST_COMPRESSED_TEST_CASE_H_

#include <Test/TestCase.h>

class CompressedTestCase : public TestCase
{
  virtual void run();
  virtual void setUp();
  void compressed();
  void compressedGradient();
  void compressedInputGradient();
  void parallelCompressed();
};

#endif // OPENANN_TEST_COMPRESSED_TEST_CASE_H_
