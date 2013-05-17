#include "SigmaPiConstraintTestCase.h"
#include <OpenANN/OpenANN>
#include <OpenANN/layers/SigmaPiConstraints.h>
#include <OpenANN/layers/SigmaPi.h>


SigmaPiConstraintTestCase::SigmaPiConstraintTestCase() : TestCase(), T1(25, 1), T2(25, 1), T3(25, 1)
{
  T1 <<
    1, 1, 1, 0, 0, 
    0, 1, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0;

  T2 << 
    0, 0, 0, 0, 0,
    0, 0, 0, 1, 0,
    0, 1, 1, 1, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 0;

  T3 << 
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 1, 1, 1,
    0, 0, 0, 1, 0,
    0, 0, 0, 1, 0;
}

void SigmaPiConstraintTestCase::run()
{
  RUN(SigmaPiConstraintTestCase, distance);
  RUN(SigmaPiConstraintTestCase, slope);
  RUN(SigmaPiConstraintTestCase, triangle);
}

void SigmaPiConstraintTestCase::distance()
{
  OpenANN::Net net;

  net.inputLayer(1, 5 ,5);

  OpenANN::DistanceConstraint constraint(5, 5);
  OpenANN::SigmaPi* layer = new OpenANN::SigmaPi(net.getOutputInfo(0), false, OpenANN::LOGISTIC, 0.05);
  layer->secondOrderNodes(1, constraint);

  net.addOutputLayer(layer);
  net.initialize();  

  Eigen::VectorXd c1 = net(T1);
  Eigen::VectorXd c2 = net(T2);
  Eigen::VectorXd c3 = net(T3);

  ASSERT(c1.size() == 1);
  ASSERT(c2.size() == 1);
  ASSERT(c3.size() == 1);

  ASSERT(c1.x() == c2.x());
  ASSERT(c2.x() == c3.x());
  ASSERT(c3.x() == c1.x());
}


void SigmaPiConstraintTestCase::slope()
{
  OpenANN::Net net;

  net.inputLayer(1, 5 ,5);

  OpenANN::SlopeConstraint constraint(5, 5);
  OpenANN::SigmaPi* layer = new OpenANN::SigmaPi(net.getOutputInfo(0), false, OpenANN::LOGISTIC, 0.05);
  layer->secondOrderNodes(1, constraint);

  net.addOutputLayer(layer);
  net.initialize();  

  Eigen::VectorXd c1 = net(T1);
  Eigen::VectorXd c3 = net(T3);

  ASSERT(c1.size() == 1);
  ASSERT(c3.size() == 1);

  ASSERT(c3.x() == c1.x());
}

void SigmaPiConstraintTestCase::triangle()
{
  OpenANN::Net net;

  net.inputLayer(1, 5 ,5);

  OpenANN::TriangleConstraint constraint(5, 5, M_PI/8);
  OpenANN::SigmaPi* layer = new OpenANN::SigmaPi(net.getOutputInfo(0), false, OpenANN::LOGISTIC, 0.05);
  layer->thirdOrderNodes(1, constraint);

  net.addOutputLayer(layer);
  net.initialize();  

  Eigen::VectorXd c1 = net(T1);
  Eigen::VectorXd c2 = net(T2);

  std::cout << "C1: " << c1 << std::endl;
  std::cout << "C2: " << c2 << std::endl;

  ASSERT(c1.size() == 1);
  ASSERT(c2.size() == 1);

  ASSERT(c2.x() == c1.x());
}

