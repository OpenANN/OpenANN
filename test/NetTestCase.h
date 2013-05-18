#pragma once

#include <Test/TestCase.h>

class NetTestCase : public TestCase
{
  virtual void run();
  void dimension();
  void error();
  void gradientSSE();
  void gradientCE();
  void multilayerNetwork();
  void predictMinibatch();
};
