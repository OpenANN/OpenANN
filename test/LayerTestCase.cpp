#include "LayerTestCase.h"
#include "FiniteDifferences.h"
#include <OpenANN/OpenANN>
#include <OpenANN/layers/FullyConnected.h>
#include <OpenANN/layers/Compressed.h>
#include <OpenANN/layers/Convolutional.h>
#include <OpenANN/layers/Subsampling.h>
#include <OpenANN/layers/MaxPooling.h>
#include <OpenANN/layers/LocalResponseNormalization.h>
#include <OpenANN/layers/SigmaPi.h>
#include <OpenANN/layers/Dropout.h>
#include <OpenANN/Learner.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/util/OpenANNException.h>

using namespace OpenANN;

class LayerOptimizable : public Learner
{
  Layer& layer;
  std::vector<double*> parameters;
  std::vector<double*> derivatives;
  OutputInfo info;
  Eigen::MatrixXd input;
  Eigen::MatrixXd desired;
public:
  LayerOptimizable(Layer& layer, OutputInfo inputs)
    : layer(layer)
  {
    info = layer.initialize(parameters, derivatives);
    input = Eigen::VectorXd::Random(inputs.outputs());
    desired = Eigen::VectorXd::Random(info.outputs());
  }

  virtual unsigned int dimension()
  {
    return parameters.size();
  }

  virtual Eigen::VectorXd currentParameters()
  {
    Eigen::VectorXd params(dimension());
    std::vector<double*>::const_iterator it = parameters.begin();
    for(int i = 0; i < dimension(); i++, it++)
      params(i) = **it;
    return params;
  }

  virtual void setParameters(const Eigen::VectorXd& parameters)
  {
    std::vector<double*>::const_iterator it = this->parameters.begin();
    for(int i = 0; i < dimension(); i++, it++)
      **it = parameters(i);
    layer.updatedParameters();
  }

  virtual double error()
  {
    Eigen::MatrixXd* output;
    layer.forwardPropagate(&input, output, false);
    double error = 0.0;
    for(int i = 0; i < desired.rows(); i++)
    {
      double diff = (*output)(i) - desired(i);
      error += diff*diff;
    }
    return error/2.0;
  }

  virtual Eigen::VectorXd gradient()
  {
    Eigen::MatrixXd* output;
    layer.forwardPropagate(&input, output, false);
    Eigen::MatrixXd diff = *output;
    for(int i = 0; i < desired.rows(); i++)
      diff(i) = (*output)(i) - desired(i);
    Eigen::MatrixXd* e;
    layer.backpropagate(&diff, e);
    Eigen::VectorXd derivs(dimension());
    std::vector<double*>::const_iterator it = derivatives.begin();
    for(int i = 0; i < dimension(); i++, it++)
      derivs(i) = **it;
    return derivs;
  }

  Eigen::VectorXd inputGradient()
  {
    Eigen::MatrixXd* output;
    layer.forwardPropagate(&input, output, false);
    Eigen::MatrixXd diff = *output;
    for(int i = 0; i < desired.rows(); i++)
      diff(i) = (*output)(i) - desired(i);
    Eigen::MatrixXd* e;
    layer.backpropagate(&diff, e);
    Eigen::VectorXd derivs(input.rows());
    for(int i = 0; i < input.rows(); i++)
      derivs(i) = (*e)(i);
    return derivs;
  }

  virtual Eigen::MatrixXd hessian()
  {
    return Eigen::MatrixXd::Random(dimension(), dimension());
  }

  virtual void initialize()
  {
  }

  virtual bool providesGradient()
  {
    return true;
  }

  virtual bool providesHessian()
  {
    return false;
  }

  virtual bool providesInitialization()
  {
    return true;
  }

  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& x)
  {
    this->input = x;
    Eigen::MatrixXd* output;
    layer.forwardPropagate(&input, output, false);
    return *output;
  }

  virtual Learner& trainingSet(Eigen::MatrixXd& trainingInput,
                               Eigen::MatrixXd& trainingOutput)
  {
    input = trainingInput.row(0);
    desired = trainingOutput.row(0);
  }

  virtual Learner& trainingSet(DataSet& trainingSet)
  {
    throw OpenANN::OpenANNException("trainingSet is not implemented in "
                                    "LayerOptimizable");
  }
};

void LayerTestCase::run()
{
  RUN(LayerTestCase, fullyConnected);
  RUN(LayerTestCase, fullyConnectedGradient);
  RUN(LayerTestCase, fullyConnectedInputGradient);
  RUN(LayerTestCase, compressed);
  RUN(LayerTestCase, compressedGradient);
  RUN(LayerTestCase, compressedInputGradient);
  RUN(LayerTestCase, convolutional);
  RUN(LayerTestCase, convolutionalGradient);
  RUN(LayerTestCase, convolutionalInputGradient);
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

void LayerTestCase::fullyConnected()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  FullyConnected layer(info, 2, false, TANH, 0.05, 0.0);

  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 2);

  for(std::vector<double*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 1.0;
  Eigen::MatrixXd x(1, 3);
  x << 0.5, 1.0, 2.0;
  Eigen::MatrixXd e(1, 3);
  e << 1.0, 2.0, 0.0;

  Eigen::MatrixXd* y = 0;
  layer.forwardPropagate(&x, y, false);
  ASSERT(y != 0);
  ASSERT_EQUALS_DELTA((*y)(0), tanh(3.5), 1e-10);
  ASSERT_EQUALS_DELTA((*y)(1), tanh(3.5), 1e-10);

  Eigen::MatrixXd* e2;
  layer.backpropagate(&e, e2);
  Eigen::VectorXd Wd(6);
  int i = 0;
  for(std::vector<double*>::iterator it = parameterDerivativePointers.begin();
      it != parameterDerivativePointers.end(); it++)
    Wd(i++) = **it;
  ASSERT_EQUALS_DELTA(Wd(0), 0.5*(1.0-(*y)(0)*(*y)(0))*1.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(1), 1.0*(1.0-(*y)(0)*(*y)(0))*1.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(2), 2.0*(1.0-(*y)(0)*(*y)(0))*1.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(3), 0.5*(1.0-(*y)(1)*(*y)(1))*2.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(4), 1.0*(1.0-(*y)(1)*(*y)(1))*2.0, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(5), 2.0*(1.0-(*y)(1)*(*y)(1))*2.0, 1e-7);
  ASSERT(e2 != 0);
}

void LayerTestCase::fullyConnectedGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  FullyConnected layer(info, 2, false, TANH, 0.05, 0.0);
  LayerOptimizable opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::parameterGradient(0, opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}

void LayerTestCase::fullyConnectedInputGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  FullyConnected layer(info, 2, false, TANH, 0.05, 0.0);
  LayerOptimizable opt(layer, info);

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(1, 3);
  Eigen::MatrixXd y = Eigen::MatrixXd::Random(1, 2);
  opt.trainingSet(x, y);
  Eigen::VectorXd gradient = opt.inputGradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::inputGradient(x.transpose(), y.transpose(), opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}

void LayerTestCase::compressed()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  Compressed layer(info, 2, 3, false, TANH, "average", 0.05);

  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers,
                                      parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 2);

  for(std::vector<double*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 1.0;
  layer.updatedParameters();
  Eigen::MatrixXd x(1, 3);
  x << 0.5, 1.0, 2.0;
  Eigen::MatrixXd e(1, 3);
  e << 1.0, 2.0, 0.0;

  Eigen::MatrixXd* y;
  layer.forwardPropagate(&x, y, false);
  ASSERT(y != 0);
  ASSERT_EQUALS_DELTA((*y)(0), tanh(3.5), 1e-10);
  ASSERT_EQUALS_DELTA((*y)(1), tanh(3.5), 1e-10);
}

void LayerTestCase::compressedGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  Compressed layer(info, 2, 2, true, TANH, "gaussian", 0.05);
  LayerOptimizable opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::parameterGradient(0, opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}

void LayerTestCase::compressedInputGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  Compressed layer(info, 2, 2, true, TANH, "gaussian", 0.05);
  LayerOptimizable opt(layer, info);

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(1, 3);
  Eigen::MatrixXd y = Eigen::MatrixXd::Random(1, 2);
  opt.trainingSet(x, y);
  Eigen::VectorXd gradient = opt.inputGradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::inputGradient(x.transpose(), y.transpose(), opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}

void LayerTestCase::convolutional()
{
  OutputInfo info;
  info.dimensions.push_back(2);
  info.dimensions.push_back(4);
  info.dimensions.push_back(4);
  Convolutional layer(info, 2, 3, 3, false, TANH, 0.05);
  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers,
                                      parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 2);
  ASSERT_EQUALS(info2.dimensions[2], 2);

  for(std::vector<double*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 0.01;
  layer.updatedParameters();

  Eigen::MatrixXd x(1, info.outputs());
  x.fill(1.0);
  Eigen::MatrixXd* y;
  layer.forwardPropagate(&x, y, false);
  ASSERT_EQUALS_DELTA((*y)(0), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(1), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(2), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(3), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(4), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(5), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(6), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(7), tanh(0.18), 1e-5);
}

void LayerTestCase::convolutionalGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(15);
  info.dimensions.push_back(15);
  Convolutional layer(info, 2, 3, 3, true, LINEAR, 0.05);
  LayerOptimizable opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::parameterGradient(0, opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-2);
}

void LayerTestCase::convolutionalInputGradient()
{
  OutputInfo info;
  info.dimensions.push_back(3);
  info.dimensions.push_back(15);
  info.dimensions.push_back(15);
  Convolutional layer(info, 2, 3, 3, true, LINEAR, 0.05);
  LayerOptimizable opt(layer, info);

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(1, 3*15*15);
  Eigen::MatrixXd y = Eigen::MatrixXd::Random(1, 2*13*13);
  opt.trainingSet(x, y);
  Eigen::VectorXd gradient = opt.inputGradient();
  Eigen::VectorXd estimatedGradient = FiniteDifferences::inputGradient(x.transpose(), y.transpose(), opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
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
  LayerOptimizable opt(layer, info);

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
  LayerOptimizable opt(layer, info);

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
  LayerOptimizable opt(layer, info);

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
  LayerOptimizable opt(layer, info);

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
  LayerOptimizable opt(layer, info);

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

  LayerOptimizable opt(layer, info);

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

  LayerOptimizable opt(layer, info);

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
  net.initialize();

  Eigen::VectorXd g = net.gradient();
  Eigen::VectorXd e = FiniteDifferences::parameterGradient(0, net);
  double delta = std::max<double>(1e-2, 1e-5*e.norm());
  for(int j = 0; j < net.dimension(); j++)
    ASSERT_EQUALS_DELTA(g(j), e(j), delta);
}
