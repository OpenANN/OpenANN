#ifndef OPENANN_TEST_IO_DATA_SET_TEST_CASE_H_
#define OPENANN_TEST_IO_DATA_SET_TEST_CASE_H_

#include <Test/TestCase.h>

class IODataSetTestCase : public TestCase
{
  virtual void run();

  void loadLibSVM();
  void saveLibSVM();
  void loadFANN();
  void saveFANN();
};

#endif // OPENANN_TEST_IO_DATA_SET_TEST_CASE_H_
