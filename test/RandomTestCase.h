#ifndef OPENANN_TEST_RANDOM_TEST_CASE_H_
#define OPENANN_TEST_RANDOM_TEST_CASE_H_

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

#endif // OPENANN_TEST_RANDOM_TEST_CASE_H_
