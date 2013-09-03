#ifndef OPENANN_TEST_CONVOLUTIONAL_TEST_CASE_H_
#define OPENANN_TEST_CONVOLUTIONAL_TEST_CASE_H_

#include <Test/TestCase.h>

class ConvolutionalTestCase : public TestCase
{
  virtual void run();
  virtual void setUp();
  void convolutional();
  void convolutionalGradient();
  void convolutionalInputGradient();
  void regularization();
};

#endif // OPENANN_TEST_CONVOLUTIONAL_TEST_CASE_H_
