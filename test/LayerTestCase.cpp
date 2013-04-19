#include "LayerTestCase.h"
#include <OpenANN/OpenANN>
#include <OpenANN/layers/FullyConnected.h>
#include <OpenANN/layers/Compressed.h>
#include <OpenANN/layers/Convolutional.h>
#include <OpenANN/layers/Subsampling.h>
#include <OpenANN/layers/MaxPooling.h>
#include <OpenANN/layers/SigmaPi.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/io/DirectStorageDataSet.h>

using namespace OpenANN;

class LayerOptimizable : public Optimizable
{
  Layer& layer;
  std::vector<double*> parameters;
  std::vector<double*> derivatives;
  OutputInfo info;
  Eigen::VectorXd input;
  Eigen::VectorXd desired;
public:
  LayerOptimizable(Layer& layer, OutputInfo inputs)
    : layer(layer)
  {
    info = layer.initialize(parameters, derivatives);
    input = Eigen::VectorXd::Random(inputs.outputs());
    desired = Eigen::VectorXd::Random(info.outputs()-info.bias);
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
    Eigen::VectorXd* output;
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
    Eigen::VectorXd* output;
    layer.forwardPropagate(&input, output, false);
    Eigen::VectorXd diff = *output;
    for(int i = 0; i < desired.rows(); i++)
      diff(i) = (*output)(i) - desired(i);
    Eigen::VectorXd* e;
    layer.backpropagate(&diff, e);
    Eigen::VectorXd derivs(dimension());
    std::vector<double*>::const_iterator it = derivatives.begin();
    for(int i = 0; i < dimension(); i++, it++)
      derivs(i) = **it;
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
};

void LayerTestCase::run()
{
  RUN(LayerTestCase, fullyConnected);
  RUN(LayerTestCase, fullyConnectedGradient);
  RUN(LayerTestCase, compressed);
  RUN(LayerTestCase, compressedGradient);
  RUN(LayerTestCase, convolutional);
  RUN(LayerTestCase, convolutionalGradient);
  RUN(LayerTestCase, subsampling);
  RUN(LayerTestCase, subsamplingGradient);
  RUN(LayerTestCase, maxPooling);
  RUN(LayerTestCase, maxPoolingGradient);
  RUN(LayerTestCase, multilayerNetwork);
  RUN(LayerTestCase, sigmaPiNoConstraintGradient);
  RUN(LayerTestCase, sigmaPiWithConstraintGradient);
}

void LayerTestCase::fullyConnected()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(3);
  FullyConnected layer(info, 2, true, TANH, 0.05, 0.0, 0.0);

  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT(info2.bias);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 3);

  for(std::vector<double*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 1.0;
  Eigen::VectorXd x(3);
  x << 0.5, 1.0, 2.0;
  Eigen::VectorXd e(3);
  e << 1.0, 2.0, 0.0;

  Eigen::VectorXd* y;
  layer.forwardPropagate(&x, y, false);
  ASSERT(y != 0);
  ASSERT_EQUALS_DELTA((*y)(0), tanh(3.5), 1e-10);
  ASSERT_EQUALS_DELTA((*y)(1), tanh(3.5), 1e-10);
  ASSERT_EQUALS_DELTA((*y)(2), 1.0, 1e-10);

  Eigen::VectorXd* e2;
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
  info.bias = false;
  info.dimensions.push_back(3);
  FullyConnected layer(info, 2, true, TANH, 0.05, 0.0, 0.0);
  LayerOptimizable opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = opt.gradientFD();
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}


void LayerTestCase::sigmaPiNoConstraintGradient()
{
    OutputInfo info;
    info.bias = false;
    info.dimensions.push_back(5);
    info.dimensions.push_back(5);
    SigmaPi layer(info, true, TANH, 0.05);
    layer.secondOrderNodes(2);

    LayerOptimizable opt(layer, info);

    Eigen::VectorXd gradient = opt.gradient();
    Eigen::VectorXd estimatedGradient = opt.gradientFD();

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
    info.bias = false;
    info.dimensions.push_back(5);
    info.dimensions.push_back(5);
    TestConstraint constraint;
    SigmaPi layer(info, true, TANH, 0.05);
    layer.secondOrderNodes(2, constraint);

    LayerOptimizable opt(layer, info);

    Eigen::VectorXd gradient = opt.gradient();
    Eigen::VectorXd estimatedGradient = opt.gradientFD();

    for(int i = 0; i < gradient.rows(); i++)
        ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-3);
}



void LayerTestCase::compressed()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(3);
  Compressed layer(info, 2, 3, true, TANH, "average", 0.05, 0.0);

  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT(info2.bias);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 3);

  for(std::vector<double*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 1.0;
  layer.updatedParameters();
  Eigen::VectorXd x(3);
  x << 0.5, 1.0, 2.0;
  Eigen::VectorXd e(3);
  e << 1.0, 2.0, 0.0;

  Eigen::VectorXd* y;
  layer.forwardPropagate(&x, y, false);
  ASSERT(y != 0);
  ASSERT_EQUALS_DELTA((*y)(0), tanh(3.5), 1e-10);
  ASSERT_EQUALS_DELTA((*y)(1), tanh(3.5), 1e-10);
  ASSERT_EQUALS_DELTA((*y)(2), 1.0, 1e-10);
}

void LayerTestCase::compressedGradient()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(3);
  Compressed layer(info, 2, 2, true, TANH, "gaussian", 0.05, 0.0);
  LayerOptimizable opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = opt.gradientFD();
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}

void LayerTestCase::convolutional()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(2);
  info.dimensions.push_back(4);
  info.dimensions.push_back(4);
  Convolutional layer(info, 2, 3, 3, true, TANH, 0.05);
  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 2);
  ASSERT_EQUALS(info2.dimensions[2], 2);

  for(std::vector<double*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 0.01;
  layer.updatedParameters();

  Eigen::VectorXd x(info.outputs());
  x.fill(1.0);
  Eigen::VectorXd* y;
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
  info.bias = false;
  info.dimensions.push_back(3);
  info.dimensions.push_back(15);
  info.dimensions.push_back(15);
  Convolutional layer(info, 2, 3, 3, true, LINEAR, 0.05);
  LayerOptimizable opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = opt.gradientFD();
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-2);
}

void LayerTestCase::subsampling()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(2);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  Subsampling layer(info, 2, 2, true, TANH, 0.05);
  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 3);
  ASSERT_EQUALS(info2.dimensions[2], 3);

  for(std::vector<double*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 0.1;

  Eigen::VectorXd x(info.outputs());
  x.fill(1.0);
  Eigen::VectorXd* y;
  layer.forwardPropagate(&x, y, false);
  for(int i = 0; i < 18; i++)
    ASSERT_EQUALS_DELTA((*y)(i), tanh(0.4), 1e-5);
}

void LayerTestCase::subsamplingGradient()
{
  OutputInfo info;
  info.bias = true;
  info.dimensions.push_back(3);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  Subsampling layer(info, 3, 3, true, LINEAR, 0.05);
  LayerOptimizable opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = opt.gradientFD();
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-4);
}

void LayerTestCase::maxPooling()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(2);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  MaxPooling layer(info, 2, 2, true);
  std::vector<double*> parameterPointers;
  std::vector<double*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 3);
  ASSERT_EQUALS(info2.dimensions[2], 3);

  Eigen::VectorXd x(info.outputs());
  x.fill(1.0);
  Eigen::VectorXd* y;
  layer.forwardPropagate(&x, y, false);
  for(int i = 0; i < 18; i++)
    ASSERT_EQUALS_DELTA((*y)(i), 1.0, 1e-5);
}

void LayerTestCase::maxPoolingGradient()
{
  OutputInfo info;
  info.bias = true;
  info.dimensions.push_back(3);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  MaxPooling layer(info, 3, 3, true);
  LayerOptimizable opt(layer, info);

  Eigen::VectorXd gradient = opt.gradient();
  Eigen::VectorXd estimatedGradient = opt.gradientFD();
}



void LayerTestCase::multilayerNetwork()
{
  int samples = 10;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(1*6*6, samples);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(3, samples);
  DirectStorageDataSet ds(X, Y);

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

  net.initialize();

  Eigen::VectorXd g = net.gradient();
  Eigen::VectorXd e = net.gradientFD();
  double delta = std::max<double>(1e-2, 1e-5*e.norm());
  for(int j = 0; j < net.dimension(); j++)
    ASSERT_EQUALS_DELTA(g(j), e(j), delta);

  Eigen::VectorXd values(samples);
  Eigen::MatrixXd gradients(samples, net.dimension());
  net.VJ(values, gradients);
  for(int n = 0; n < samples; n++)
  {
    Eigen::VectorXd e = net.singleGradientFD(n);
    double delta = std::max<double>(1e-2, 1e-5*e.norm());
    for(int j = 0; j < net.dimension(); j++)
      ASSERT_EQUALS_DELTA(gradients(n, j), e(j), delta);
    double error = net.error(n);
    ASSERT_EQUALS_DELTA(values(n), error, 1e-2);
  }
}
