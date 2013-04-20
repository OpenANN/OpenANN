#pragma once

#include <Test/TestCase.h>

class RandomTestCase : public TestCase
{
  virtual void run();
  void seed();
  void generateInt();
  void generateIndex();
  void generate();
  void sampleNormalDistribution();
  void generateIndices();
};
