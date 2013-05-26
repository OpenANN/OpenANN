#include <Test/TestCase.h>

class RBMTestCase : public TestCase
{
  virtual void run();
  virtual void setUp();
  void learnSimpleExample();
  void parameterGradient();
  void inputGradient();
};
