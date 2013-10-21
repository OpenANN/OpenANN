#include "BaggingTestCase.h"
#include <OpenANN/Bagging.h>
#include <OpenANN/Net.h>
#include <OpenANN/Evaluation.h>
#include <OpenANN/optimization/CG.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/util/Random.h>

void BaggingTestCase::run()
{
  RUN(BaggingTestCase, bagging);
}

void BaggingTestCase::setUp()
{
  OpenANN::RandomNumberGenerator().seed(0);
}

void BaggingTestCase::bagging()
{
  const int D = 1;
  const int F = 1;
  const int N = 100;
  const int models = 10;
  Eigen::MatrixXd X(N, D);
  OpenANN::RandomNumberGenerator rng;
  for(int n = 0; n < N; n++)
  {
    X(n, 0) = 2*M_PI * (double) n / (double) (N-1) + rng.sampleNormalDistribution<double>() * 0.3;
  }
  Eigen::MatrixXd T(N, F);
  T.array() = X.array().cos();
  OpenANN::DirectStorageDataSet dataSet(&X, &T);

  OpenANN::Bagging bagging(0.2);

  std::list<OpenANN::Net*> nets;
  for(int m = 0; m < models; m++)
  {
    OpenANN::Net* net = new OpenANN::Net;
    net->inputLayer(D);
    net->fullyConnectedLayer(2, OpenANN::LOGISTIC);
    net->outputLayer(F, OpenANN::LINEAR);
    nets.push_back(net);
    bagging.addLearner(*net);
  }

  OpenANN::CG optimizer;
  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 100;
  optimizer.setStopCriteria(stop);
  bagging.setOptimizer(optimizer);
  bagging.train(dataSet);
  Eigen::MatrixXd Y = bagging(X);
  const double mse = (Y - T).squaredNorm() / N;

  // Combined classifier should be better than single classifiers
  double averageSse = 0.0;
  for(std::list<OpenANN::Net*>::iterator it = nets.begin();
      it != nets.end(); it++)
    averageSse += OpenANN::mse(**it, dataSet);
  averageSse /= (double) models;
  ASSERT_WITHIN(averageSse, 0, 0.1);
  ASSERT_WITHIN(mse, 0, averageSse);

  for(std::list<OpenANN::Net*>::iterator it = nets.begin();
      it != nets.end(); it++)
    delete *it;
}
