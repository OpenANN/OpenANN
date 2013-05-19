#include "MaxPoolingTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/layers/MaxPooling.h>
#include <OpenANN/util/Random.h>

void MaxPoolingTestCase::run()
{
  RUN(MaxPoolingTestCase, maxPooling);
  RUN(MaxPoolingTestCase, maxPoolingInputGradient);
}

void MaxPoolingTestCase::setUp()
{
  OpenANN::RandomNumberGenerator rng;
  rng.seed(0);
}

void MaxPoolingTestCase::maxPooling()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(2);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  OpenANN::MaxPooling layer(info, 2, 2);
  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OpenANN::OutputInfo info2 = layer.initialize(parameterPointers,
                                      parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 3);
  ASSERT_EQUALS(info2.dimensions[2], 3);

  Eigen::MatrixXd x(1, info.outputs());
  x.fill(1.0);
  Eigen::MatrixXd* y;
  layer.forwardPropagate(&x, y, false);
  for(int i = 0; i < 18; i++)
    ASSERT_EQUALS_DELTA((*y)(i), 1.0, 1e-5);
}

void MaxPoolingTestCase::maxPoolingInputGradient()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(2);
  info.dimensions.push_back(4);
  info.dimensions.push_back(4);
  OpenANN::MaxPooling layer(info, 2, 2);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(1, 2*4*4);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(1, 2*2*2);
  opt.trainingSet(X, Y);
  Eigen::MatrixXd gradient = opt.inputGradient();
  Eigen::MatrixXd estimatedGradient = OpenANN::FiniteDifferences::inputGradient(
      X, Y, opt, 1e-5);
  for(int j = 0; j < gradient.rows(); j++)
    for(int i = 0; i < gradient.cols(); i++)
      ASSERT_EQUALS_DELTA(gradient(j, i), estimatedGradient(j, i), 1e-10);
}
