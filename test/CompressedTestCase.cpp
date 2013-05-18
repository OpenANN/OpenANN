#include "CompressedTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/layers/Compressed.h>

using namespace OpenANN;

void CompressedTestCase::run()
{
  RUN(CompressedTestCase, compressed);
  RUN(CompressedTestCase, compressedGradient);
  RUN(CompressedTestCase, compressedInputGradient);
  RUN(CompressedTestCase, parallelCompressed);
}

void CompressedTestCase::compressed()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  Compressed layer(info, 2, 3, false, TANH, "average", 0.05);

  std::vector<double*> pp;
  std::vector<double*> pdp;
  OutputInfo info2 = layer.initialize(pp, pdp);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 2);

  for(std::vector<double*>::iterator it = pp.begin(); it != pp.end(); it++)
    **it = 1.0;
  layer.updatedParameters();
  Eigen::MatrixXd x(1, 3);
  x << 0.5, 1.0, 2.0;
  Eigen::MatrixXd e(1, 2);
  e << 1.0, 2.0;

  Eigen::MatrixXd* y;
  layer.forwardPropagate(&x, y, false);
  ASSERT(y != 0);
  ASSERT_EQUALS_DELTA((*y)(0, 0), tanh(3.5), 1e-10);
  ASSERT_EQUALS_DELTA((*y)(0, 1), tanh(3.5), 1e-10);
}

void CompressedTestCase::compressedGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  Compressed layer(info, 2, 2, true, TANH, "gaussian", 0.05);
  LayerAdapter opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::parameterGradient(
      0, opt, 1e-5);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-10);
}

void CompressedTestCase::compressedInputGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  Compressed layer(info, 2, 2, true, TANH, "gaussian", 0.05);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(1, 3);
  Eigen::MatrixXd y = Eigen::MatrixXd::Random(1, 2);
  opt.trainingSet(x, y);
  Eigen::VectorXd gradient = opt.inputGradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::inputGradient(
      x.transpose(), y.transpose(), opt, 1e-5);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-10);
}

void CompressedTestCase::parallelCompressed()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  Compressed layer(info, 2, 3, false, TANH, "average", 0.05);

  std::vector<double*> pp;
  std::vector<double*> pdp;
  OutputInfo info2 = layer.initialize(pp, pdp);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 2);

  for(std::vector<double*>::iterator it = pp.begin(); it != pp.end(); it++)
    **it = 1.0;
  layer.updatedParameters();
  Eigen::MatrixXd x(2, 3);
  x << 0.5, 1.0, 2.0,
       0.5, 1.0, 2.0;
  Eigen::MatrixXd e(2, 2);
  e << 1.0, 2.0,
       1.0, 2.0;

  Eigen::MatrixXd* y;
  layer.forwardPropagate(&x, y, false);
  ASSERT(y != 0);
  ASSERT_EQUALS(y->rows(), 2);
  ASSERT_EQUALS(y->cols(), 2);
  ASSERT_EQUALS_DELTA((*y)(0, 0), tanh(3.5), 1e-10);
  ASSERT_EQUALS_DELTA((*y)(0, 1), tanh(3.5), 1e-10);

  Eigen::MatrixXd* e2;
  layer.backpropagate(&e, e2);
  Eigen::VectorXd Wd(8);
  int i = 0;
  for(std::vector<double*>::iterator it = pdp.begin(); it != pdp.end(); it++)
    Wd(i++) = **it;
}
