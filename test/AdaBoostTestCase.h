#ifndef OPENANN_TEST_ADA_BOOST_TEST_CASE_H_
#define OPENANN_TEST_ADA_BOOST_TEST_CASE_H_

#include <Test/TestCase.h>

class AdaBoostTestCase : public TestCase
{
  virtual void run();
  void adaBoost();
};

#endif // OPENANN_TEST_ADA_BOOST_TEST_CASE_H_
