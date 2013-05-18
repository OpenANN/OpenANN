#pragma

#include <Test/TestCase.h>

class MaxPoolingTestCase : public TestCase
{
  virtual void run();
  virtual void setUp();
  void maxPooling();
  void maxPoolingGradient();
  void maxPoolingInputGradient();
};
