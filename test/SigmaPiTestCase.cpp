#include "SigmaPiTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/OpenANN>
#include <OpenANN/layers/SigmaPi.h>

using namespace OpenANN;

void SigmaPiTestCase::run()
{
  RUN(SigmaPiTestCase, sigmaPiNoConstraintGradient);
  RUN(SigmaPiTestCase, sigmaPiWithConstraintGradient);
}

void SigmaPiTestCase::sigmaPiNoConstraintGradient()
{
  OutputInfo info;
  info.dimensions.push_back(5);
  info.dimensions.push_back(5);
  SigmaPi layer(info, false, TANH, 0.05);
  layer.secondOrderNodes(2);

  LayerAdapter opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::parameterGradient(0, opt);

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
  OutputInfo info;
  info.dimensions.push_back(5);
  info.dimensions.push_back(5);
  TestConstraint constraint;
  SigmaPi layer(info, false, TANH, 0.05);
  layer.secondOrderNodes(2, constraint);

  LayerAdapter opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::parameterGradient(0, opt);

  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-2);
}
