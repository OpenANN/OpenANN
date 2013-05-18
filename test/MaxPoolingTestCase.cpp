#include "MaxPoolingTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/layers/MaxPooling.h>
#include <OpenANN/util/Random.h>

using namespace OpenANN;

void MaxPoolingTestCase::run()
{
  RUN(MaxPoolingTestCase, maxPooling);
  RUN(MaxPoolingTestCase, maxPoolingInputGradient);
}

void MaxPoolingTestCase::setUp()
{
  RandomNumberGenerator rng;
  rng.seed(0);
}

void MaxPoolingTestCase::maxPooling()
{
  OutputInfo info;
  info.dimensions.push_back(2);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  MaxPooling layer(info, 2, 2);
  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers,
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
  OutputInfo info;
  info.dimensions.push_back(2);
  info.dimensions.push_back(4);
  info.dimensions.push_back(4);
  MaxPooling layer(info, 2, 2);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(1, 2*4*4);
  Eigen::MatrixXd y = Eigen::MatrixXd::Random(1, 2*2*2);
  opt.trainingSet(x, y);
  Eigen::VectorXd gradient = opt.inputGradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::inputGradient(
      x.transpose(), y.transpose(), opt, 1e-5);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-10);
}
