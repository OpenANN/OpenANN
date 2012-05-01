#include "Test/TestCase.h"

class MLPImplementationTestCase : public TestCase
{
  virtual void run();
  void uncompressedForwardPropagation();
  void compressedForwardPropagation();
};