#include "AdaBoostTestCase.h"
#include <OpenANN/AdaBoost.h>
#include <OpenANN/Net.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/optimization/LMA.h>
#include <list>

void AdaBoostTestCase::run()
{
  RUN(AdaBoostTestCase, adaBoost);
}

//#include <iostream>
void AdaBoostTestCase::adaBoost()
{
  OpenANN::AdaBoost adaBoost;

  std::list<OpenANN::Net*> nets;
  for(int m = 0; m < 5; m++)
  {
    OpenANN::Net* net = new OpenANN::Net;
    net->inputLayer(2);
    net->outputLayer(1, OpenANN::TANH);
    nets.push_back(net);
    adaBoost.addLearner(*net);
  }

  OpenANN::LMA optimizer;
  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 10;
  optimizer.setStopCriteria(stop);
  adaBoost.setOptimizer(optimizer);

  const int D = 2;
  const int F = 1;
  const int N = 4;
  Eigen::MatrixXd X(N, D);
  Eigen::MatrixXd T(N, F);
  X.row(0) << 0.0, 1.0;
  T.row(0) << 1.0;
  X.row(1) << 0.0, 0.0;
  T.row(1) << 0.0;
  X.row(2) << 1.0, 1.0;
  T.row(2) << 0.0;
  X.row(3) << 1.0, 0.0;
  T.row(3) << 1.0;
  OpenANN::DirectStorageDataSet dataSet(&X, &T);
  adaBoost.train(dataSet);

  //std::cout << adaBoost(X) << std::endl;

  for(std::list<OpenANN::Net*>::iterator it = nets.begin();
      it != nets.end(); it++)
    delete *it;
}
