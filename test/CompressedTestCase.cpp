#include "CompressedTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/layers/Compressed.h>
#include <OpenANN/util/Random.h>

void CompressedTestCase::run()
{
  RUN(CompressedTestCase, compressed);
  RUN(CompressedTestCase, compressedGradient);
  RUN(CompressedTestCase, compressedInputGradient);
  RUN(CompressedTestCase, parallelCompressed);
}

void CompressedTestCase::setUp()
{
  OpenANN::RandomNumberGenerator rng;
  rng.seed(0);
}

void CompressedTestCase::compressed()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  OpenANN::Compressed layer(info, 2, 3, false, OpenANN::TANH, "average", 0.05,
                            OpenANN::Regularization());

  std::vector<double*> pp;
  std::vector<double*> pdp;
  OpenANN::OutputInfo info2 = layer.initialize(pp, pdp);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 2);

  for(std::vector<double*>::iterator it = pp.begin(); it != pp.end(); ++it)
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
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  OpenANN::Compressed layer(info, 2, 2, true, OpenANN::TANH, "gaussian", 0.05,
                            OpenANN::Regularization());
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

void CompressedTestCase::compressedInputGradient()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  OpenANN::Compressed layer(info, 2, 2, true, OpenANN::TANH, "gaussian", 0.05,
                            OpenANN::Regularization());
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 3);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 2);
  opt.trainingSet(X, Y);
  Eigen::MatrixXd gradient = opt.inputGradient();
  Eigen::MatrixXd estimatedGradient = OpenANN::FiniteDifferences::
      inputGradient(X, Y, opt, 1e-5);
  for(int j = 0; j < gradient.rows(); j++)
    for(int i = 0; i < gradient.cols(); i++)
      ASSERT_EQUALS_DELTA(gradient(j, i), estimatedGradient(j, i), 1e-10);
}

void CompressedTestCase::parallelCompressed()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  OpenANN::Compressed layer(info, 2, 3, false, OpenANN::TANH, "average", 0.05,
                            OpenANN::Regularization());

  std::vector<double*> pp;
  std::vector<double*> pdp;
  OpenANN::OutputInfo info2 = layer.initialize(pp, pdp);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 2);

  for(std::vector<double*>::iterator it = pp.begin(); it != pp.end(); ++it)
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

  double error = 0;
  Eigen::MatrixXd* e2;
  layer.backpropagate(&e, e2, true, error);
  Eigen::VectorXd Wd(8);
  int i = 0;
  for(std::vector<double*>::iterator it = pdp.begin(); it != pdp.end(); ++it)
    Wd(i++) = **it;
}
