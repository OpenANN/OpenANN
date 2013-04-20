#pragma once

#include "Test/TestCase.h"

class LayerTestCase : public TestCase
{
  virtual void run();
  void fullyConnected();
  void fullyConnectedGradient();
  void fullyConnectedInputGradient();
  void compressed();
  void compressedGradient();
  void compressedInputGradient();
  void convolutional();
  void convolutionalGradient();
  void subsampling();
  void subsamplingGradient();
  void maxPooling();
  void maxPoolingGradient();
  void sigmaPiNoConstraintGradient();
  void sigmaPiWithConstraintGradient();
  void multilayerNetwork();
};
