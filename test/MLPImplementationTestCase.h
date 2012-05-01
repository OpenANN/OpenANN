#include "Test/TestCase.h"

class MLPImplementationTestCase : public TestCase
{
  virtual void run();
  void twoDimensionalOrthogonalFunctionsMatrix();
  void uncompressedForwardPropagation();
  void compressedForwardPropagation();
};