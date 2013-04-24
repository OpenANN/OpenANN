#include "EvaluationTestCase.h"
#include <OpenANN/Evaluation.h>

class RandomLearner : public OpenANN::Learner
{
  int F;
public:
  RandomLearner(int F)
    : F(F)
  {
  }

  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& x)
  {
    return Eigen::VectorXd::Random(F);
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
}

void EvaluationTestCase::sse()
{
  
}
