#include "Test/TestCase.h"

class MLPTestCase : public TestCase
{
  virtual void run();
  void uncompressedBackpropagation();
  void compressedBackpropagation();
  void uncompressedBackpropagationWithoutBias();
  void compressedBackpropagationWithoutBias();
  void finishIterationWithoutDataSet();
  void network1Backprop();
};