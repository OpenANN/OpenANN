#include "RBMTestCase.h"
#include "LayerAdapter.h"
#include "FiniteDifferences.h"
#include <OpenANN/RBM.h>
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <Eigen/Dense>

void RBMTestCase::run()
{
  RUN(RBMTestCase, learnSimpleExample);
  RUN(RBMTestCase, parameterGradient);
  RUN(RBMTestCase, inputGradient);
}

void RBMTestCase::setUp()
{
  OpenANN::RandomNumberGenerator rng;
  rng.seed(2);
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

  OpenANN::RBM rbm(6, 2, 1, 0.1);
  OpenANN::DirectStorageDataSet ds(&X);
  rbm.trainingSet(ds);
  rbm.initialize();
  OpenANN::MBSGD opt(0.1, 0.0, 2);
  OpenANN::StoppingCriteria stop;
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

void RBMTestCase::parameterGradient()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  OpenANN::RBM layer(3, 2, 1, 0.01, true);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 3);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 2);
  std::vector<int> indices;
  indices.push_back(0);
  indices.push_back(1);
  opt.trainingSet(X, Y);
  Eigen::VectorXd gradient = opt.gradient(indices.begin(), indices.end());
  Eigen::VectorXd estimatedGradient = OpenANN::FiniteDifferences::
                                      parameterGradient(indices.begin(), indices.end(), opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-10);
}

void RBMTestCase::inputGradient()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  OpenANN::RBM layer(3, 2, 1, 0.01, 0.0, true);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 3);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 2);
  opt.trainingSet(X, Y);
  Eigen::MatrixXd gradient = opt.inputGradient();
  ASSERT_EQUALS(gradient.rows(), 2);
  Eigen::MatrixXd estimatedGradient = OpenANN::FiniteDifferences::
                                      inputGradient(X, Y, opt);
  ASSERT_EQUALS(estimatedGradient.rows(), 2);
  for(int j = 0; j < gradient.rows(); j++)
    for(int i = 0; i < gradient.cols(); i++)
      ASSERT_EQUALS_DELTA(gradient(j, i), estimatedGradient(j, i), 1e-10);
}
