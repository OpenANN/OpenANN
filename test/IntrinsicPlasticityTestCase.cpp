#include "IntrinsicPlasticityTestCase.h"
#include "LayerAdapter.h"
#include "FiniteDifferences.h"
#include <OpenANN/IntrinsicPlasticity.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/io/DirectStorageDataSet.h>

void IntrinsicPlasticityTestCase::run()
{
  RUN(IntrinsicPlasticityTestCase, learn);
  RUN(IntrinsicPlasticityTestCase, backprop);
}

void IntrinsicPlasticityTestCase::setUp()
{
  OpenANN::RandomNumberGenerator rng;
  rng.seed(0);
}

void IntrinsicPlasticityTestCase::learn()
{
  OpenANN::IntrinsicPlasticity ip(2, 0.2);

  OpenANN::RandomNumberGenerator rng;
  int samples = 10000;
  Eigen::MatrixXd X(samples, 2);
  Eigen::MatrixXd Y(samples, 2);
  for(int i = 0; i < samples; i++)
  {
    for(int j = 0; j < 2; j++)
    {
      X(i, j) = rng.sampleNormalDistribution<double>();
      Y(i, j) = rng.sampleNormalDistribution<double>();
    }
  }
  OpenANN::DirectStorageDataSet ds(&X, &Y);
  ip.trainingSet(ds);

  Eigen::VectorXd p = ip.currentParameters();
  ASSERT_EQUALS_DELTA(p(0), 1.0, 1e-3);
  ASSERT_EQUALS_DELTA(p(1), 1.0, 1e-3);
  ASSERT_NOT_EQUALS(p(2), 0.0);
  ASSERT_NOT_EQUALS(p(3), 0.0);
  p(2) = 1e-3;
  p(3) = 1e-3;
  ip.setParameters(p);
  p = ip.currentParameters();
  ASSERT_EQUALS_DELTA(p(0), 1.0, 1e-3);
  ASSERT_EQUALS_DELTA(p(1), 1.0, 1e-3);
  ASSERT_NOT_EQUALS(p(2), 0.0);
  ASSERT_NOT_EQUALS(p(3), 0.0);

  Eigen::VectorXd y(2);
  y.fill(0.0);
  for(int i = 0; i < samples; i++)
    y += ip(ds.getInstance(i));
  Eigen::VectorXd mean = y / (double) samples;
  ASSERT_EQUALS_DELTA(mean(0), 0.5, 2e-2);
  ASSERT_EQUALS_DELTA(mean(1), 0.5, 2e-2);
  const double e = ip.error();

  OpenANN::MBSGD sgd(5e-5, 0.9, 1);
  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 1;
  sgd.setOptimizable(ip);
  sgd.setStopCriteria(stop);
  while(sgd.step());

  y.fill(0.0);
  for(int i = 0; i < samples; i++)
    y += ip(ds.getInstance(i));
  mean = y / (double) samples;
  ASSERT_EQUALS_DELTA(mean(0), 0.2, 2e-2);
  ASSERT_EQUALS_DELTA(mean(1), 0.2, 2e-2);
  ASSERT(e > ip.error());
}

void IntrinsicPlasticityTestCase::backprop()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(5);
  OpenANN::IntrinsicPlasticity layer(5, 0.2);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 5);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 5);
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
