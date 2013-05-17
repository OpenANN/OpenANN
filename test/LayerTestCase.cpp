#include "LayerTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/OpenANN>
#include <OpenANN/layers/Subsampling.h>
#include <OpenANN/layers/MaxPooling.h>
#include <OpenANN/layers/LocalResponseNormalization.h>
#include <OpenANN/layers/SigmaPi.h>
#include <OpenANN/layers/Dropout.h>
#include <OpenANN/io/DirectStorageDataSet.h>

using namespace OpenANN;

void LayerTestCase::run()
{
  RUN(LayerTestCase, subsampling);
  RUN(LayerTestCase, subsamplingGradient);
  RUN(LayerTestCase, subsamplingInputGradient);
  RUN(LayerTestCase, maxPooling);
  RUN(LayerTestCase, maxPoolingGradient);
  RUN(LayerTestCase, maxPoolingInputGradient);
  RUN(LayerTestCase, localResponseNormalizationInputGradient);
  RUN(LayerTestCase, dropout);
  RUN(LayerTestCase, sigmaPiNoConstraintGradient);
  RUN(LayerTestCase, sigmaPiWithConstraintGradient);
  RUN(LayerTestCase, multilayerNetwork);
}

void LayerTestCase::subsampling()
{
  OutputInfo info;
  info.dimensions.push_back(2);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  Subsampling layer(info, 2, 2, false, TANH, 0.05);
  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers,
                                      parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 3);
  ASSERT_EQUALS(info2.dimensions[2], 3);

  for(std::vector<double*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 0.1;

  Eigen::MatrixXd x(1, info.outputs());
  x.fill(1.0);
  Eigen::MatrixXd* y;
  layer.forwardPropagate(&x, y, false);
  for(int i = 0; i < 18; i++)
    ASSERT_EQUALS_DELTA((*y)(i), tanh(0.4), 1e-5);
}

void LayerTestCase::subsamplingGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  Subsampling layer(info, 3, 3, true, LINEAR, 0.05);
  LayerAdapter opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::parameterGradient(0, opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}

void LayerTestCase::subsamplingInputGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  Subsampling layer(info, 3, 3, true, LINEAR, 0.05);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(1, 3*6*6);
  Eigen::MatrixXd y = Eigen::MatrixXd::Random(1, 3*2*2);
  opt.trainingSet(x, y);
  Eigen::VectorXd gradient = opt.inputGradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::inputGradient(x.transpose(), y.transpose(), opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}

void LayerTestCase::maxPooling()
{
  OutputInfo info;
  info.dimensions.push_back(2);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  MaxPooling layer(info, 2, 2);
  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers,
                                      parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 3);
  ASSERT_EQUALS(info2.dimensions[2], 3);

  Eigen::MatrixXd x(1, info.outputs());
  x.fill(1.0);
  Eigen::MatrixXd* y;
  layer.forwardPropagate(&x, y, false);
  for(int i = 0; i < 18; i++)
    ASSERT_EQUALS_DELTA((*y)(i), 1.0, 1e-5);
}

void LayerTestCase::maxPoolingGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  MaxPooling layer(info, 3, 3);
  LayerAdapter opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::parameterGradient(0, opt);
}

void LayerTestCase::maxPoolingInputGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  MaxPooling layer(info, 3, 3);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(1, 3*6*6);
  Eigen::MatrixXd y = Eigen::MatrixXd::Random(1, 3*2*2);
  opt.trainingSet(x, y);
  Eigen::VectorXd gradient = opt.inputGradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::inputGradient(x.transpose(), y.transpose(), opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}

void LayerTestCase::localResponseNormalizationInputGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(3);
  info.dimensions.push_back(3);
  LocalResponseNormalization layer(info, 1, 3, 1e-5, 0.75);
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(1, 3*3*3);
  Eigen::MatrixXd y = Eigen::MatrixXd::Random(1, 3*3*3);
  opt.trainingSet(x, y);
  Eigen::VectorXd gradient = opt.inputGradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::inputGradient(x.transpose(), y.transpose(), opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}

void LayerTestCase::dropout()
{
  double dropoutProbability = 0.5;
  int samples = 10000;
  OutputInfo info;
  info.dimensions.push_back(samples);
  Dropout layer(info, 0.5);
  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers,
                                      parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.dimensions[0], samples);

  // During training (dropout = true) approximately dropoutProbability neurons
  // should be suppressed
  Eigen::MatrixXd x(1, samples);
  x.fill(1.0);
  Eigen::MatrixXd* y;
  layer.forwardPropagate(&x, y, true);
  double mean = y->sum() / samples;
  ASSERT_EQUALS_DELTA(mean, 0.5, 0.01);
  // After training, the output should be scaled down
  layer.forwardPropagate(&x, y, false);
  mean = y->sum() / samples;
  ASSERT_EQUALS(mean, 0.5);
}

void LayerTestCase::sigmaPiNoConstraintGradient()
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

void LayerTestCase::sigmaPiWithConstraintGradient()
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

void LayerTestCase::multilayerNetwork()
{
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(1, 1*6*6);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(1, 3);
  DirectStorageDataSet ds(&X, &Y);

  Net net;
  net.inputLayer(1, 6, 6);
  net.convolutionalLayer(4, 3, 3, TANH, 0.5);
  net.localReponseNormalizationLayer(2.0, 3, 0.01, 0.75);
  net.subsamplingLayer(2, 2, TANH, 0.5);
  net.fullyConnectedLayer(10, TANH, 0.5);
  net.extremeLayer(10, TANH, 0.05);
  net.outputLayer(3, LINEAR, 0.5);
  net.trainingSet(ds);

  Eigen::VectorXd g = net.gradient();
  Eigen::VectorXd e = FiniteDifferences::parameterGradient(0, net);
  double delta = std::max<double>(1e-2, 1e-5*e.norm());
  for(int j = 0; j < net.dimension(); j++)
    ASSERT_EQUALS_DELTA(g(j), e(j), delta);
}
