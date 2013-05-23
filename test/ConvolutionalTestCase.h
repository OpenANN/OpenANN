#pragma once

#include <Test/TestCase.h>

class ConvolutionalTestCase : public TestCase
{
  virtual void run();
  virtual void setUp();
  void convolutional();
  void convolutionalGradient();
  void convolutionalInputGradient();
};
