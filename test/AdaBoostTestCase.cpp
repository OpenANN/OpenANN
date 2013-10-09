#include "AdaBoostTestCase.h"
#include <OpenANN/AdaBoost.h>
#include <OpenANN/Net.h>
#include <OpenANN/Evaluation.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/optimization/CG.h>
#include <OpenANN/util/Random.h>
#include <list>

void AdaBoostTestCase::run()
{
  RUN(AdaBoostTestCase, adaBoost);
}

void AdaBoostTestCase::setUp()
{
  OpenANN::RandomNumberGenerator().seed(0);
}

void AdaBoostTestCase::adaBoost()
{
  OpenANN::AdaBoost adaBoost;

  std::list<OpenANN::Net*> nets;
  for(int m = 0; m < 4; m++)
  {
    OpenANN::Net* net = new OpenANN::Net;
    net->inputLayer(2);
    net->outputLayer(1, OpenANN::LOGISTIC);
    nets.push_back(net);
    adaBoost.addLearner(*net);
  }

  OpenANN::CG optimizer;
  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 100;
  optimizer.setStopCriteria(stop);
  adaBoost.setOptimizer(optimizer);

  const int D = 2;
  const int F = 1;
  const int N = 100;
  Eigen::MatrixXd X(N, D);
  Eigen::MatrixXd T(N, F);
  OpenANN::RandomNumberGenerator rng;
  for(int n = 0; n < N; n++)
  {
    X(n, 0) = rng.sampleNormalDistribution<double>();
    X(n, 1) = rng.sampleNormalDistribution<double>();
    T(n, 0) = (double) (X.row(n).squaredNorm() > 1.0);
  }
  OpenANN::DirectStorageDataSet dataSet(&X, &T);
  adaBoost.train(dataSet);
  Eigen::MatrixXd Y = adaBoost(X);
  const int correct = ((Y - T).array() < 0.5).count();
  ASSERT_WITHIN(correct, 80, 100);

  // Combined classifier should be better than single classifiers
  for(std::list<OpenANN::Net*>::iterator it = nets.begin();
      it != nets.end(); it++)
    ASSERT_WITHIN(OpenANN::classificationHits(**it, dataSet), 51, correct);

  for(std::list<OpenANN::Net*>::iterator it = nets.begin();
      it != nets.end(); it++)
    delete *it;
}
