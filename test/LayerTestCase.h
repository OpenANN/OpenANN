#pragma once

#include "Test/TestCase.h"

class LayerTestCase : public TestCase
{
  virtual void run();
  void fullyConnected();
  void fullyConnectedGradient();
  void compressed();
  void compressedGradient();
  void convolutional();
  void convolutionalGradient();
  void subsampling();
  void subsamplingGradient();
  void maxPooling();
  void maxPoolingGradient();
  void multilayerNetwork();
  void sigmaPiNoConstraintGradient();
  void sigmaPiWithConstraintGradient();
};
