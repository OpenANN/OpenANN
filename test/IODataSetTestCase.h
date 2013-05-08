#pragma once

#include <Test/TestCase.h>

class IODataSetTestCase : public TestCase
{
  virtual void run();

  void loadLibSVM();
  void saveLibSVM();
  void loadFANN();
  void saveFANN();
};
