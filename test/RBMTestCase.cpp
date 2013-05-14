#include "RBMTestCase.h"
#include <OpenANN/RBM.h>
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <Eigen/Dense>

using namespace OpenANN;

void RBMTestCase::run()
{
  RUN(RBMTestCase, learnSimpleExample);
}

void RBMTestCase::setUp()
{
  OpenANN::RandomNumberGenerator rng;
  rng.seed(0);
}

void RBMTestCase::learnSimpleExample()
{
  Eigen::MatrixXd X(6, 6);
  X.row(0) << 1, 1, 1, 0, 0, 0;
  X.row(1) << 1, 0, 1, 0, 0, 0;
  X.row(2) << 1, 1, 1, 0, 0, 0;
  X.row(3) << 0, 0, 1, 1, 1, 0;
  X.row(4) << 0, 0, 1, 1, 0, 0;
  X.row(5) << 0, 0, 1, 1, 1, 0;

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

  Eigen::MatrixXd H = rbm(X);

  for(int i = 0; i < 3; i++)
  {
    Eigen::MatrixXd v = rbm.reconstructProb(i, 1);
    ASSERT(v(0, 0) > 0.5);
    ASSERT(v(0, 1) > 0.5);
    ASSERT(v(0, 2) > 0.5);
    ASSERT(v(0, 3) < 0.5);
    ASSERT(v(0, 4) < 0.5);
    ASSERT(v(0, 5) < 0.5);

    ASSERT(H(i, 0) > 0.5);
    ASSERT(H(i, 1) < 0.5);
  }

  for(int i = 3; i < 6; i++)
  {
    Eigen::MatrixXd v = rbm.reconstructProb(i, 1);
    ASSERT(v(0, 0) < 0.5);
    ASSERT(v(0, 1) < 0.5);
    ASSERT(v(0, 2) > 0.5);
    ASSERT(v(0, 3) > 0.5);
    ASSERT(v(0, 4) > 0.5);
    ASSERT(v(0, 5) < 0.5);

    ASSERT(H(i, 0) < 0.5);
    ASSERT(H(i, 1) > 0.5);
  }
}
