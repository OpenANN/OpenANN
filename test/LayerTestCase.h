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
  void convolutionalInputGradient();
  void subsampling();
  void subsamplingGradient();
  void subsamplingInputGradient();
  void maxPooling();
  void maxPoolingGradient();
  void maxPoolingInputGradient();
  void localResponseNormalizationInputGradient();
  void dropout();
  void sigmaPiNoConstraintGradient();
  void sigmaPiWithConstraintGradient();
  void multilayerNetwork();
};
