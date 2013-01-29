#include "Test/TestCase.h"

class LayerTestCase : public TestCase
{
  virtual void run();
  void fullyConnected();
  void fullyConnectedGradient();
  void convolutional();
  void convolutionalGradient();
  void subsampling();
  void subsamplingGradient();
};