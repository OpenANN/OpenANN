#include "LocalResponseNormalizationTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/layers/LocalResponseNormalization.h>

using namespace OpenANN;

void LocalResponseNormalizationTestCase::run()
{
  RUN(LocalResponseNormalizationTestCase, localResponseNormalizationInputGradient);
}

void LocalResponseNormalizationTestCase::localResponseNormalizationInputGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(3);
  info.dimensions.push_back(3);
  LocalResponseNormalization layer(info, 1, 3, 1e-5, 0.75);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 3*3*3);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 3*3*3);
  opt.trainingSet(X, Y);
  Eigen::MatrixXd gradient = opt.inputGradient();
  ASSERT_EQUALS(gradient.rows(), 2);
  Eigen::MatrixXd estimatedGradient = FiniteDifferences::inputGradient(X, Y,
                                                                       opt);
  for(int j = 0; j < gradient.rows(); j++)
    for(int i = 0; i < gradient.cols(); i++)
      ASSERT_EQUALS_DELTA(gradient(j, i), estimatedGradient(j, i), 1e-4);
}
