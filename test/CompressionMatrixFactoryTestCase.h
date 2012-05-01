#pragma once

#include "Test/TestCase.h"

class CompressionMatrixFactoryTestCase : public TestCase
{
  virtual void run();

  void compress1D();
  void compress2D();
  void compress3D();
};