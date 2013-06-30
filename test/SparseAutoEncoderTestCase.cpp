#include "SparseAutoEncoderTestCase.h"
#include "FiniteDifferences.h"
#include <OpenANN/SparseAutoEncoder.h>
#include <OpenANN/io/DirectStorageDataSet.h>

void SparseAutoEncoderTestCase::run()
{
  RUN(SparseAutoEncoderTestCase, gradient);
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
