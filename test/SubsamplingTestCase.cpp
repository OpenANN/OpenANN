#include "SubsamplingTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/layers/Subsampling.h>

using namespace OpenANN;

void SubsamplingTestCase::run()
{
  RUN(SubsamplingTestCase, subsampling);
  RUN(SubsamplingTestCase, subsamplingGradient);
  RUN(SubsamplingTestCase, subsamplingInputGradient);
}

void SubsamplingTestCase::subsampling()
{
  OutputInfo info;
  info.dimensions.push_back(2);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  Subsampling layer(info, 2, 2, false, TANH, 0.05);
  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers,
                                      parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 3);
  ASSERT_EQUALS(info2.dimensions[2], 3);

  for(std::vector<double*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 0.1;

  Eigen::MatrixXd x(1, info.outputs());
  x.fill(1.0);
  Eigen::MatrixXd* y;
  layer.forwardPropagate(&x, y, false);
  for(int i = 0; i < 18; i++)
    ASSERT_EQUALS_DELTA((*y)(i), tanh(0.4), 1e-5);
}

void SubsamplingTestCase::subsamplingGradient()
{
  OutputInfo info;
  info.dimensions.push_back(2);
  info.dimensions.push_back(4);
  info.dimensions.push_back(4);
  Subsampling layer(info, 2, 2, true, LINEAR, 0.05);
  LayerAdapter opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::parameterGradient(
      0, opt, 1e-5);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-10);
}

void SubsamplingTestCase::subsamplingInputGradient()
{
  OutputInfo info;
  info.dimensions.push_back(2);
  info.dimensions.push_back(4);
  info.dimensions.push_back(4);
  Subsampling layer(info, 2, 2, true, LINEAR, 0.05);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(1, 2*4*4);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(1, 2*2*2);
  opt.trainingSet(X, Y);
  Eigen::MatrixXd gradient = opt.inputGradient();
  Eigen::MatrixXd estimatedGradient = FiniteDifferences::inputGradient(
      X, Y, opt, 1e-5);
  for(int j = 0; j < gradient.rows(); j++)
    for(int i = 0; i < gradient.cols(); i++)
      ASSERT_EQUALS_DELTA(gradient(j, i), estimatedGradient(j, i), 1e-10);
}
