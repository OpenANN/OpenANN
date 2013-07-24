#ifndef OPENANN_TEST_NET_TEST_CASE_H_
#define OPENANN_TEST_NET_TEST_CASE_H_

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
  void minibatchErrorGradient();
  void saveLoad();
};

#endif // OPENANN_TEST_NET_TEST_CASE_H_
