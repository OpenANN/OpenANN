#ifndef OPENANN_TEST_KMEANS_TEST_CASE_H_
#define OPENANN_TEST_KMEANS_TEST_CASE_H_

#include "Test/TestCase.h"

class KMeansTestCase : public TestCase
{
  virtual void run();
  void clustering();
};

#endif // OPENANN_TEST_KMEANS_TEST_CASE_H_
