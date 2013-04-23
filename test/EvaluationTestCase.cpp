#include "EvaluationTestCase.h"
#include <OpenANN/Evaluation.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/util/Random.h>

class RandomLearner : public OpenANN::Learner
{
  OpenANN::RandomNumberGenerator rng;
  int F;
  bool normalize;
public:
  RandomLearner(int F, bool normalize)
    : F(F), normalize(normalize)
  {
  }

  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& x)
  {
    Eigen::VectorXd res(F);
    if(normalize)
    {
      for(int f = 0; f < F; f++)
        res(f) = rng.generate<double>(0.0, 1.0);
      res /= res.sum();
    }
    else
    {
      for(int f = 0; f < F; f++)
        res(f) = rng.sampleNormalDistribution<double>()*2.0;
    }
    return res;
  }

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
  const int N = 1000;
  const int D = 1;
  const int F = 10;
  Eigen::MatrixXd X(D, N);
  Eigen::MatrixXd Y(F, N);
  Y.fill(0.0);
  RandomLearner learner(F, false);
  OpenANN::DirectStorageDataSet dataSet(X, Y);
  double sse = OpenANN::sse(learner, dataSet);
  ASSERT_EQUALS_DELTA((int) sse, F*N*2*2, 1000);
}

void EvaluationTestCase::mse()
{
  const int N = 1000;
  const int D = 1;
  const int F = 10;
  Eigen::MatrixXd X(D, N);
  Eigen::MatrixXd Y(F, N);
  Y.fill(0.0);
  RandomLearner learner(F, false);
  OpenANN::DirectStorageDataSet dataSet(X, Y);
  double mse = OpenANN::mse(learner, dataSet);
  ASSERT_EQUALS_DELTA(mse, F*2.0*2.0, 1.0);
}

void EvaluationTestCase::rmse()
{
  const int N = 100000;
  const int D = 1;
  const int F = 10;
  Eigen::MatrixXd X(D, N);
  Eigen::MatrixXd Y(F, N);
  Y.fill(0.0);
  RandomLearner learner(F, false);
  OpenANN::DirectStorageDataSet dataSet(X, Y);
  double rmse = OpenANN::rmse(learner, dataSet);
  ASSERT_EQUALS_DELTA(rmse, std::sqrt(F*2.0*2.0), 0.5);
}

void EvaluationTestCase::ce()
{
  const int N = 100000;
  const int D = 1;
  const int F = 5;
  Eigen::MatrixXd X(D, N);
  Eigen::MatrixXd Y(F, N);
  Y.fill(0.0);
  OpenANN::RandomNumberGenerator rng;
  for(int n = 0; n < N; n++)
    Y(rng.generateIndex(F), n) = 1.0;
  RandomLearner learner(F, true);
  OpenANN::DirectStorageDataSet dataSet(X, Y);
  double ce = OpenANN::ce(learner, dataSet);
  ASSERT_EQUALS_DELTA(ce, -N*std::log(0.5/F), N/10.0);
}
