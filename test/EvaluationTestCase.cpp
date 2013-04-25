#include "EvaluationTestCase.h"
#include <OpenANN/Learner.h>
#include <OpenANN/Evaluation.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/util/Random.h>

class ReturnInput : public OpenANN::Learner
{
public:
  ReturnInput() {}
  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& x) { return x; }
  virtual Learner& trainingSet(Eigen::MatrixXd& trainingInput, Eigen::MatrixXd& trainingOutput) {}
  virtual Learner& trainingSet(OpenANN::DataSet& trainingSet) {}
  virtual Eigen::VectorXd currentParameters() {}
  virtual unsigned int dimension() {}
  virtual double error() {}
  virtual Eigen::VectorXd gradient() {}
  virtual Eigen::MatrixXd hessian() {}
  virtual void initialize() {}
  virtual bool providesGradient() {}
  virtual bool providesHessian() {}
  virtual bool providesInitialization() {}
  virtual void setParameters(const Eigen::VectorXd& parameters) {}
};

void EvaluationTestCase::run()
{
  RUN(EvaluationTestCase, sse);
  RUN(EvaluationTestCase, mse);
  RUN(EvaluationTestCase, rmse);
  RUN(EvaluationTestCase, ce);
}

void EvaluationTestCase::sse()
{
  OpenANN::RandomNumberGenerator rng;
  const int N = 1000;
  const int D = 10;
  const int F = 10;
  Eigen::MatrixXd Y(D, N);
  for(int n = 0; n < N; n++)
    for(int f = 0; f < F; f++)
      Y(f, n) = rng.sampleNormalDistribution<double>()*2.0;
  Eigen::MatrixXd T(F, N);
  T.fill(0.0);
  ReturnInput learner;
  OpenANN::DirectStorageDataSet dataSet(Y, T);
  double sse = OpenANN::sse(learner, dataSet);
  ASSERT_EQUALS_DELTA((int) sse, F*N*2*2, 1000);
}

void EvaluationTestCase::mse()
{
  OpenANN::RandomNumberGenerator rng;
  const int N = 1000;
  const int D = 10;
  const int F = 10;
  Eigen::MatrixXd Y(D, N);
  for(int n = 0; n < N; n++)
    for(int f = 0; f < F; f++)
      Y(f, n) = rng.sampleNormalDistribution<double>()*2.0;
  Eigen::MatrixXd T(F, N);
  T.fill(0.0);
  ReturnInput learner;
  OpenANN::DirectStorageDataSet dataSet(Y, T);
  double mse = OpenANN::mse(learner, dataSet);
  ASSERT_EQUALS_DELTA(mse, F*2.0*2.0, 1.0);
}

void EvaluationTestCase::rmse()
{
  OpenANN::RandomNumberGenerator rng;
  const int N = 100000;
  const int D = 10;
  const int F = 10;
  Eigen::MatrixXd Y(D, N);
  for(int n = 0; n < N; n++)
    for(int f = 0; f < F; f++)
      Y(f, n) = rng.sampleNormalDistribution<double>()*2.0;
  Eigen::MatrixXd T(F, N);
  T.fill(0.0);
  ReturnInput learner;
  OpenANN::DirectStorageDataSet dataSet(Y, T);
  double rmse = OpenANN::rmse(learner, dataSet);
  ASSERT_EQUALS_DELTA(rmse, std::sqrt(F*2.0*2.0), 0.5);
}

void EvaluationTestCase::ce()
{
  const int N = 2;
  const int D = 2;
  const int F = 2;
  Eigen::MatrixXd Y(D, N);
  Eigen::MatrixXd T(F, N);
  Y.col(0) << 0.5, 0.5;
  Y.col(1) << 0.0, 1.0;
  T.col(0) << 0.0, 1.0;
  T.col(1) << 1.0, 0.0;
  ReturnInput learner;
  OpenANN::DirectStorageDataSet dataSet(Y, T);
  double ce = OpenANN::ce(learner, dataSet);
  ASSERT_EQUALS_DELTA(ce, 23.72, 0.01);
}
