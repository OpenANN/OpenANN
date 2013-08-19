#include "SparseAutoEncoderTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/SparseAutoEncoder.h>
#include <OpenANN/io/DirectStorageDataSet.h>

void SparseAutoEncoderTestCase::run()
{
  RUN(SparseAutoEncoderTestCase, gradient);
  RUN(SparseAutoEncoderTestCase, inputGradient);
  RUN(SparseAutoEncoderTestCase, layerGradient);
}

void SparseAutoEncoderTestCase::gradient()
{
  const int D = 6;
  const int H = 3;
  const int N = 1;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  OpenANN::DirectStorageDataSet ds(&X);
  OpenANN::SparseAutoEncoder sae(D, H, 3.0, 0.1, 0.0001, OpenANN::LOGISTIC);
  sae.trainingSet(ds);
  Eigen::VectorXd ga = OpenANN::FiniteDifferences::parameterGradient(0, sae);
  Eigen::VectorXd g = sae.gradient();
  for(int k = 0; k < sae.dimension(); k++)
    ASSERT_EQUALS_DELTA(ga(k), g(k), 1e-2);
}

void SparseAutoEncoderTestCase::inputGradient()
{
  const int D = 6;
  const int H = 3;
  const int N = 2;
  OpenANN::OutputInfo info;
  info.dimensions.push_back(D);
  OpenANN::SparseAutoEncoder layer(D, H, 3.0, 0.1, 0.0001, OpenANN::LOGISTIC);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(N, H);
  opt.trainingSet(X, Y);
  Eigen::MatrixXd gradient = opt.inputGradient();
  ASSERT_EQUALS(gradient.rows(), 2);
  Eigen::MatrixXd estimatedGradient =
      OpenANN::FiniteDifferences::inputGradient(X, Y, opt);
  ASSERT_EQUALS(estimatedGradient.rows(), 2);
  for(int j = 0; j < gradient.rows(); j++)
    for(int i = 0; i < gradient.cols(); i++)
      ASSERT_EQUALS_DELTA(gradient(j, i), estimatedGradient(j, i), 1e-10);
}

void SparseAutoEncoderTestCase::layerGradient()
{
  // TODO implement
  ASSERT(false);
}
