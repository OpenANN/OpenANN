#include "FullyConnectedTestCase.h"
#include "LayerAdapter.h"
#include "FiniteDifferences.h"
#include <OpenANN/layers/FullyConnected.h>

using namespace OpenANN;

void FullyConnectedTestCase::run()
{
  RUN(FullyConnectedTestCase, forward);
  RUN(FullyConnectedTestCase, backprop);
  RUN(FullyConnectedTestCase, inputGradient);
  RUN(FullyConnectedTestCase, parallelForward);
}

void FullyConnectedTestCase::forward()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  FullyConnected layer(info, 2, false, TANH, 0.05, OpenANN::Regularization());

  std::vector<double*> pp;
  std::vector<double*> pdp;
  OutputInfo info2 = layer.initialize(pp, pdp);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 2);

  for(std::vector<double*>::iterator it = pp.begin(); it != pp.end(); ++it)
    **it = 1.0;
  Eigen::MatrixXd x(1, 3);
  x << 0.5, 1.0, 2.0;
  Eigen::MatrixXd e(1, 2);
  e << 1.0, 2.0;

  Eigen::MatrixXd* y = 0;
  layer.forwardPropagate(&x, y, false);
  ASSERT(y != 0);
  ASSERT_EQUALS_DELTA((*y)(0, 0), tanh(3.5), 1e-10);
  ASSERT_EQUALS_DELTA((*y)(0, 1), tanh(3.5), 1e-10);

  Eigen::MatrixXd* e2;
  layer.backpropagate(&e, e2, true);
  Eigen::VectorXd Wd(6);
  int i = 0;
  for(std::vector<double*>::iterator it = pdp.begin(); it != pdp.end(); ++it)
    Wd(i++) = **it;
  ASSERT_EQUALS_DELTA(Wd(0), 0.5*(1.0-(*y)(0)*(*y)(0))*1.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(1), 1.0*(1.0-(*y)(0)*(*y)(0))*1.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(2), 2.0*(1.0-(*y)(0)*(*y)(0))*1.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(3), 0.5*(1.0-(*y)(1)*(*y)(1))*2.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(4), 1.0*(1.0-(*y)(1)*(*y)(1))*2.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(5), 2.0*(1.0-(*y)(1)*(*y)(1))*2.0, 1e-7);
  ASSERT(e2 != 0);
}

void FullyConnectedTestCase::backprop()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  FullyConnected layer(info, 2, false, TANH, 0.05, OpenANN::Regularization());
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 3);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 2);
  std::vector<int> indices;
  indices.push_back(0);
  indices.push_back(1);
  opt.trainingSet(X, Y);
  Eigen::VectorXd gradient = opt.gradient(indices.begin(), indices.end());
  Eigen::VectorXd estimatedGradient = FiniteDifferences::parameterGradient(
      indices.begin(), indices.end(), opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-10);
}

void FullyConnectedTestCase::inputGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  FullyConnected layer(info, 2, false, TANH, 0.05, OpenANN::Regularization());
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 3);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 2);
  opt.trainingSet(X, Y);
  Eigen::MatrixXd gradient = opt.inputGradient();
  ASSERT_EQUALS(gradient.rows(), 2);
  Eigen::MatrixXd estimatedGradient = FiniteDifferences::inputGradient(X, Y,
                                                                       opt);
  ASSERT_EQUALS(estimatedGradient.rows(), 2);
  for(int j = 0; j < gradient.rows(); j++)
    for(int i = 0; i < gradient.cols(); i++)
      ASSERT_EQUALS_DELTA(gradient(j, i), estimatedGradient(j, i), 1e-10);
}

void FullyConnectedTestCase::parallelForward()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  FullyConnected layer(info, 2, true, TANH, 0.05, OpenANN::Regularization());

  std::vector<double*> pp;
  std::vector<double*> pdp;
  OutputInfo info2 = layer.initialize(pp, pdp);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 2);

  for(std::vector<double*>::iterator it = pp.begin(); it != pp.end(); ++it)
    **it = 1.0;
  Eigen::MatrixXd x(2, 3);
  x << 0.5, 1.0, 2.0,
       0.5, 1.0, 2.0;
  Eigen::MatrixXd e(2, 2);
  e << 1.0, 2.0,
       1.0, 2.0;

  Eigen::MatrixXd* y = 0;
  layer.forwardPropagate(&x, y, false);
  ASSERT(y != 0);
  ASSERT_EQUALS_DELTA((*y)(0, 0), tanh(4.5), 1e-10);
  ASSERT_EQUALS_DELTA((*y)(0, 1), tanh(4.5), 1e-10);
  ASSERT_EQUALS_DELTA((*y)(1, 0), tanh(4.5), 1e-10);
  ASSERT_EQUALS_DELTA((*y)(1, 1), tanh(4.5), 1e-10);

  Eigen::MatrixXd* e2;
  layer.backpropagate(&e, e2, true);
  Eigen::VectorXd Wd(8);
  int i = 0;
  for(std::vector<double*>::iterator it = pdp.begin(); it != pdp.end(); ++it)
    Wd(i++) = **it;
  ASSERT_EQUALS_DELTA(Wd(0), 2.0*0.5*(1.0-(*y)(0, 0)*(*y)(0, 0))*1.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(1), 2.0*1.0*(1.0-(*y)(0, 0)*(*y)(0, 0))*1.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(2), 2.0*2.0*(1.0-(*y)(0, 0)*(*y)(0, 0))*1.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(3), 2.0*1.0*(1.0-(*y)(0, 0)*(*y)(0, 0))*1.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(4), 2.0*0.5*(1.0-(*y)(0, 1)*(*y)(0, 1))*2.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(5), 2.0*1.0*(1.0-(*y)(0, 1)*(*y)(0, 1))*2.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(6), 2.0*2.0*(1.0-(*y)(0, 1)*(*y)(0, 1))*2.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(7), 2.0*1.0*(1.0-(*y)(0, 1)*(*y)(0, 1))*2.0, 1e-7);
  ASSERT(e2 != 0);
}
