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
  DirectStorageDataSet ds(X, X);
  rbm.trainingSet(ds);
  rbm.initialize();
  MBSGD opt(0.1, 0.0, 2);
  StoppingCriteria stop;
  stop.maximalIterations = 500;
  opt.setOptimizable(rbm);
  opt.setStopCriteria(stop);
  opt.optimize();

  for(int i = 0; i < 6; i++)
  {
    OPENANN_DEBUG << rbm.reconstructProb(i, 1).transpose();
    OPENANN_DEBUG << rbm(X.col(i)).transpose();
  }
}
