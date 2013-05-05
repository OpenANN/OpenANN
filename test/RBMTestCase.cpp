#include "RBMTestCase.h"
#include <OpenANN/RBM.h>
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <Eigen/Dense>

using namespace OpenANN;

void RBMTestCase::run()
{
  OpenANN::RandomNumberGenerator rng;
  rng.seed(0);
  RUN(RBMTestCase, learnSimpleExample);
}

void RBMTestCase::learnSimpleExample()
{
  Eigen::MatrixXd X(6, 6);
  X.col(0) << 1, 1, 1, 0, 0, 0;
  X.col(1) << 1, 0, 1, 0, 0, 0;
  X.col(2) << 1, 1, 1, 0, 0, 0;
  X.col(3) << 0, 0, 1, 1, 1, 0;
  X.col(4) << 0, 0, 1, 1, 0, 0;
  X.col(5) << 0, 0, 1, 1, 1, 0;

  RBM rbm(6, 2, 1, 0.1);
  DirectStorageDataSet ds(&X);
  rbm.trainingSet(ds);
  rbm.initialize();
  MBSGD opt(0.1, 0.0, 2);
  StoppingCriteria stop;
  stop.maximalIterations = 2000;
  opt.setOptimizable(rbm);
  opt.setStopCriteria(stop);
  opt.optimize();

  for(int i = 0; i < 3; i++)
  {
    Eigen::VectorXd v = rbm.reconstructProb(i, 1);
    ASSERT(v(0) > 0.5);
    ASSERT(v(1) > 0.5);
    ASSERT(v(2) > 0.5);
    ASSERT(v(3) < 0.5);
    ASSERT(v(4) < 0.5);
    ASSERT(v(5) < 0.5);
    Eigen::VectorXd h = rbm(X.col(i));
    ASSERT(h(0) > 0.5);
    ASSERT(h(1) < 0.5);
  }

  for(int i = 3; i < 6; i++)
  {
    Eigen::VectorXd v = rbm.reconstructProb(i, 1);
    ASSERT(v(0) < 0.5);
    ASSERT(v(1) < 0.5);
    ASSERT(v(2) > 0.5);
    ASSERT(v(3) > 0.5);
    ASSERT(v(4) > 0.5);
    ASSERT(v(5) < 0.5);
    Eigen::VectorXd h = rbm(X.col(i));
    ASSERT(h(0) < 0.5);
    ASSERT(h(1) > 0.5);
  }
}
