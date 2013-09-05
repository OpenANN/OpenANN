#include "ConvolutionalTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/layers/Convolutional.h>
#include <OpenANN/util/Random.h>

void ConvolutionalTestCase::run()
{
  RUN(ConvolutionalTestCase, convolutional);
  RUN(ConvolutionalTestCase, convolutionalGradient);
  RUN(ConvolutionalTestCase, convolutionalInputGradient);
  RUN(ConvolutionalTestCase, regularization);
}

void ConvolutionalTestCase::setUp()
{
  OpenANN::RandomNumberGenerator rng;
  rng.seed(6);
}

void ConvolutionalTestCase::convolutional()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(2);
  info.dimensions.push_back(4);
  info.dimensions.push_back(4);
  OpenANN::Convolutional layer(info, 2, 3, 3, false, OpenANN::TANH, 0.05, OpenANN::Regularization());
  std::vector<double*> pp;
  std::vector<double*> pdp;
  OpenANN::OutputInfo info2 = layer.initialize(pp, pdp);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 2);
  ASSERT_EQUALS(info2.dimensions[2], 2);

  for(std::vector<double*>::iterator it = pp.begin(); it != pp.end(); ++it)
    **it = 0.01;
  layer.updatedParameters();

  Eigen::MatrixXd x(1, info.outputs());
  x.fill(1.0);
  Eigen::MatrixXd* y;
  layer.forwardPropagate(&x, y, false);
  ASSERT_EQUALS_DELTA((*y)(0), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(1), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(2), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(3), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(4), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(5), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(6), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(7), tanh(0.18), 1e-5);
}

void ConvolutionalTestCase::convolutionalGradient()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(5);
  info.dimensions.push_back(5);
  OpenANN::Convolutional layer(info, 2, 3, 3, true, OpenANN::LINEAR, 0.05, OpenANN::Regularization());
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 3*5*5);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 2*3*3);
  std::vector<int> indices;
  indices.push_back(0);
  indices.push_back(1);
  opt.trainingSet(X, Y);
  Eigen::VectorXd gradient = opt.gradient(indices.begin(), indices.end());
  Eigen::VectorXd estimatedGradient = OpenANN::FiniteDifferences::parameterGradient(
      indices.begin(), indices.end(), opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-10);
}

void ConvolutionalTestCase::convolutionalInputGradient()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(5);
  info.dimensions.push_back(5);
  OpenANN::Convolutional layer(info, 2, 3, 3, true, OpenANN::LINEAR, 0.05, OpenANN::Regularization());
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 3*5*5);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 2*3*3);
  std::vector<int> indices;
  indices.push_back(0);
  indices.push_back(1);
  opt.trainingSet(X, Y);
  Eigen::MatrixXd gradient = opt.inputGradient();
  ASSERT_EQUALS(gradient.rows(), 2);
  Eigen::MatrixXd estimatedGradient = OpenANN::FiniteDifferences::inputGradient(X, Y,
                                                                       opt);
  ASSERT_EQUALS(estimatedGradient.rows(), 2);
  for(int j = 0; j < gradient.rows(); j++)
    for(int i = 0; i < gradient.cols(); i++)
      ASSERT_EQUALS_DELTA(gradient(j, i), estimatedGradient(j, i), 1e-10);
}

void ConvolutionalTestCase::regularization()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(5);
  info.dimensions.push_back(5);
  OpenANN::Convolutional layer(info, 2, 3, 3, true, OpenANN::LINEAR, 0.05,
                               OpenANN::Regularization(0.1, 0.1));
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(1, 3*5*5);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(1, 2*3*3);
  opt.trainingSet(X, Y);
  Eigen::VectorXd gradient = opt.gradient(0);
  Eigen::VectorXd estimatedGradient = OpenANN::FiniteDifferences::parameterGradient(0, opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-10);
}
