#include "ConvolutionalTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/layers/Convolutional.h>

using namespace OpenANN;

void ConvolutionalTestCase::run()
{
  RUN(ConvolutionalTestCase, convolutional);
  RUN(ConvolutionalTestCase, convolutionalGradient);
  RUN(ConvolutionalTestCase, convolutionalInputGradient);
}

void ConvolutionalTestCase::convolutional()
{
  OutputInfo info;
  info.dimensions.push_back(2);
  info.dimensions.push_back(4);
  info.dimensions.push_back(4);
  Convolutional layer(info, 2, 3, 3, false, TANH, 0.05);
  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers,
                                      parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 2);
  ASSERT_EQUALS(info2.dimensions[2], 2);

  for(std::vector<double*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
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
  OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(5);
  info.dimensions.push_back(5);
  Convolutional layer(info, 2, 3, 3, true, LINEAR, 0.05);
  LayerAdapter opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::parameterGradient(0, opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-2);
}

void ConvolutionalTestCase::convolutionalInputGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(5);
  info.dimensions.push_back(5);
  Convolutional layer(info, 2, 3, 3, true, LINEAR, 0.05);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(1, 3*5*5);
  Eigen::MatrixXd y = Eigen::MatrixXd::Random(1, 2*3*3);
  opt.trainingSet(x, y);
  Eigen::VectorXd gradient = opt.inputGradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::inputGradient(x.transpose(), y.transpose(), opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}
