#ifndef OPENANN_TEST_PREPROCESSING_TEST_CASE_H_
#define OPENANN_TEST_PREPROCESSING_TEST_CASE_H_

#include <Test/TestCase.h>

class PreprocessingTestCase : public TestCase
{
  virtual void run();
  void scaling();
};

#endif // OPENANN_TEST_PREPROCESSING_TEST_CASE_H_
