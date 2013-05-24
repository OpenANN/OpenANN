#include "SigmaPiTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/OpenANN>
#include <OpenANN/layers/SigmaPi.h>

void SigmaPiTestCase::run()
{
  RUN(SigmaPiTestCase, sigmaPiNoConstraintGradient);
  RUN(SigmaPiTestCase, sigmaPiWithConstraintGradient);
}

void SigmaPiTestCase::sigmaPiNoConstraintGradient()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(5);
  info.dimensions.push_back(5);
  OpenANN::SigmaPi layer(info, false, OpenANN::TANH, 0.05);
  layer.secondOrderNodes(2);

  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 5*5);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 2);
  std::vector<int> indices;
  indices.push_back(0);
  indices.push_back(1);
  opt.trainingSet(X, Y);
  Eigen::VectorXd gradient = opt.gradient(indices.begin(), indices.end());
  Eigen::VectorXd estimatedGradient = OpenANN::FiniteDifferences::
      parameterGradient(indices.begin(), indices.end(), opt);

  for(int i = 0; i < gradient.rows(); i++)
      ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}


struct TestConstraint : public OpenANN::SigmaPi::Constraint
{
  virtual double operator() (int p1, int p2) const {
    double x1 = p1 % 5;
    double y1 = p1 / 5;
    double x2 = p2 % 5;
    double y2 = p2 / 5;

    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
  }
};

void SigmaPiTestCase::sigmaPiWithConstraintGradient()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(5);
  info.dimensions.push_back(5);
  TestConstraint constraint;
  OpenANN::SigmaPi layer(info, false, OpenANN::TANH, 0.05);
  layer.secondOrderNodes(2, constraint);

  LayerAdapter opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = OpenANN::FiniteDifferences::parameterGradient(0, opt);

  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-2);
}
