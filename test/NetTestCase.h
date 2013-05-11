#pragma once

#include <Test/TestCase.h>

class NetTestCase : public TestCase
{
  virtual void run();
  virtual void dimension();
  virtual void error();
  virtual void gradientSSE();
  virtual void gradientCE();
};