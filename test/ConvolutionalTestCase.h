#pragma once

#include <Test/TestCase.h>

class ConvolutionalTestCase : public TestCase
{
  virtual void run();
  void convolutional();
  void convolutionalGradient();
  void convolutionalInputGradient();
};
