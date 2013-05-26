#pragma once

#include <Test/TestCase.h>

class CompressedTestCase : public TestCase
{
  virtual void run();
  void compressed();
  void compressedGradient();
  void compressedInputGradient();
  void parallelCompressed();
};
