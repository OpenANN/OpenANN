#include "LayerTestCase.h"
#include <layers/FullyConnected.h>
#include <layers/Compressed.h>
#include <layers/Convolutional.h>
#include <layers/Subsampling.h>
#include <layers/MaxPooling.h>
#include <Optimizable.h>
#include <DeepNetwork.h>
#include <io/DirectStorageDataSet.h>

using namespace OpenANN;

class LayerOptimizable : public Optimizable
{
  Layer& layer;
  std::vector<fpt*> parameters;
  std::vector<fpt*> derivatives;
  OutputInfo info;
  Vt input;
  Vt desired;
public:
  LayerOptimizable(Layer& layer, OutputInfo inputs)
    : layer(layer)
  {
    info = layer.initialize(parameters, derivatives);
    input = Vt::Random(inputs.outputs());
    desired = Vt::Random(info.outputs()-info.bias);
  }

  virtual unsigned int dimension()
  {
    return parameters.size();
  }

  virtual Vt currentParameters()
  {
    Vt params(dimension());
    std::vector<fpt*>::const_iterator it = parameters.begin();
    for(int i = 0; i < dimension(); i++, it++)
      params(i) = **it;
    return params;
  }

  virtual void setParameters(const Vt& parameters)
  {
    std::vector<fpt*>::const_iterator it = this->parameters.begin();
    for(int i = 0; i < dimension(); i++, it++)
      **it = parameters(i);
    layer.updatedParameters();
  }

  virtual fpt error()
  {
    Vt* output;
    layer.forwardPropagate(&input, output, false);
    fpt error = 0.0;
    for(int i = 0; i < desired.rows(); i++)
    {
      fpt diff = (*output)(i) - desired(i);
      error += diff*diff;
    }
    return error/2.0;
  }

  virtual Vt gradient()
  {
    Vt* output;
    layer.forwardPropagate(&input, output, false);
    Vt diff = *output;
    for(int i = 0; i < desired.rows(); i++)
      diff(i) = (*output)(i) - desired(i);
    Vt* e;
    layer.backpropagate(&diff, e);
    Vt derivs(dimension());
    std::vector<fpt*>::const_iterator it = derivatives.begin();
    for(int i = 0; i < dimension(); i++, it++)
      derivs(i) = **it;
    return derivs;
  }

  virtual Mt hessian()
  {
    return Mt::Random(dimension(), dimension());
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
}

void LayerTestCase::fullyConnected()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(3);
  FullyConnected layer(info, 2, true, TANH, 0.05, 0.0);

  std::vector<fpt*> parameterPointers;
  std::vector<fpt*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT(info2.bias);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 3);

  for(std::vector<fpt*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 1.0;
  Vt x(3);
  x << 0.5, 1.0, 2.0;
  Vt e(3);
  e << 1.0, 2.0, 0.0;

  Vt* y;
  layer.forwardPropagate(&x, y, false);
  ASSERT(y != 0);
  ASSERT_EQUALS_DELTA((*y)(0), (fpt) tanh(3.5), (fpt) 1e-10);
  ASSERT_EQUALS_DELTA((*y)(1), (fpt) tanh(3.5), (fpt) 1e-10);
  ASSERT_EQUALS_DELTA((*y)(2), (fpt) 1.0, (fpt) 1e-10);

  Vt* e2;
  layer.backpropagate(&e, e2);
  Vt Wd(6);
  int i = 0;
  for(std::vector<fpt*>::iterator it = parameterDerivativePointers.begin();
      it != parameterDerivativePointers.end(); it++)
    Wd(i++) = **it;
  ASSERT_EQUALS_DELTA(Wd(0), (fpt) (0.5*(1.0-(*y)(0)*(*y)(0))*1.0), (fpt) 1e-7);
  ASSERT_EQUALS_DELTA(Wd(1), (fpt) (1.0*(1.0-(*y)(0)*(*y)(0))*1.0), (fpt) 1e-7);
  ASSERT_EQUALS_DELTA(Wd(2), (fpt) (2.0*(1.0-(*y)(0)*(*y)(0))*1.0), (fpt) 1e-7);
  ASSERT_EQUALS_DELTA(Wd(3), (fpt) (0.5*(1.0-(*y)(1)*(*y)(1))*2.0), (fpt) 1e-7);
  ASSERT_EQUALS_DELTA(Wd(4), (fpt) (1.0*(1.0-(*y)(1)*(*y)(1))*2.0), (fpt) 1e-7);
  ASSERT_EQUALS_DELTA(Wd(5), (fpt) (2.0*(1.0-(*y)(1)*(*y)(1))*2.0), (fpt) 1e-7);
  ASSERT(e2 != 0);
}

void LayerTestCase::fullyConnectedGradient()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(3);
  FullyConnected layer(info, 2, true, TANH, 0.05, 0.0);
  LayerOptimizable opt(layer, info);

  Vt gradient = opt.gradient();
  Vt estimatedGradient = opt.gradientFD();
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), (fpt) 1e-4);
}

void LayerTestCase::compressed()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(3);
  Compressed layer(info, 2, 3, true, TANH, "average", 0.05, 0.0);

  std::vector<fpt*> parameterPointers;
  std::vector<fpt*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT(info2.bias);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 3);

  for(std::vector<fpt*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 1.0;
  layer.updatedParameters();
  Vt x(3);
  x << 0.5, 1.0, 2.0;
  Vt e(3);
  e << 1.0, 2.0, 0.0;

  Vt* y;
  layer.forwardPropagate(&x, y, false);
  ASSERT(y != 0);
  ASSERT_EQUALS_DELTA((*y)(0), (fpt) tanh(3.5), (fpt) 1e-10);
  ASSERT_EQUALS_DELTA((*y)(1), (fpt) tanh(3.5), (fpt) 1e-10);
  ASSERT_EQUALS_DELTA((*y)(2), (fpt) 1.0, (fpt) 1e-10);
}

void LayerTestCase::compressedGradient()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(3);
  Compressed layer(info, 2, 2, true, TANH, "gaussian", 0.05, 0.0);
  LayerOptimizable opt(layer, info);

  Vt gradient = opt.gradient();
  Vt estimatedGradient = opt.gradientFD();
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), (fpt) 1e-4);
}

void LayerTestCase::convolutional()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(2);
  info.dimensions.push_back(4);
  info.dimensions.push_back(4);
  Convolutional layer(info, 2, 3, 3, true, TANH, 0.05);
  std::vector<fpt*> parameterPointers;
  std::vector<fpt*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 2);
  ASSERT_EQUALS(info2.dimensions[2], 2);

  for(std::vector<fpt*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 0.01;
  layer.updatedParameters();

  Vt x(info.outputs());
  x.fill(1.0);
  Vt* y;
  layer.forwardPropagate(&x, y, false);
  ASSERT_EQUALS_DELTA((*y)(0), (fpt) tanh(0.18), (fpt) 1e-5);
  ASSERT_EQUALS_DELTA((*y)(1), (fpt) tanh(0.18), (fpt) 1e-5);
  ASSERT_EQUALS_DELTA((*y)(2), (fpt) tanh(0.18), (fpt) 1e-5);
  ASSERT_EQUALS_DELTA((*y)(3), (fpt) tanh(0.18), (fpt) 1e-5);
  ASSERT_EQUALS_DELTA((*y)(4), (fpt) tanh(0.18), (fpt) 1e-5);
  ASSERT_EQUALS_DELTA((*y)(5), (fpt) tanh(0.18), (fpt) 1e-5);
  ASSERT_EQUALS_DELTA((*y)(6), (fpt) tanh(0.18), (fpt) 1e-5);
  ASSERT_EQUALS_DELTA((*y)(7), (fpt) tanh(0.18), (fpt) 1e-5);
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

  Vt gradient = opt.gradient();
  Vt estimatedGradient = opt.gradientFD();
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), (fpt) 1e-2);
}

void LayerTestCase::subsampling()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(2);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  Subsampling layer(info, 2, 2, true, TANH, 0.05);
  std::vector<fpt*> parameterPointers;
  std::vector<fpt*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 3);
  ASSERT_EQUALS(info2.dimensions[2], 3);

  for(std::vector<fpt*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 0.1;

  Vt x(info.outputs());
  x.fill(1.0);
  Vt* y;
  layer.forwardPropagate(&x, y, false);
  for(int i = 0; i < 18; i++)
    ASSERT_EQUALS_DELTA((*y)(i), (fpt) tanh(0.4), (fpt) 1e-5);
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

  Vt gradient = opt.gradient();
  Vt estimatedGradient = opt.gradientFD();
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), (fpt) 1e-4);
}

void LayerTestCase::maxPooling()
{
  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(2);
  info.dimensions.push_back(6);
  info.dimensions.push_back(6);
  MaxPooling layer(info, 2, 2, true);
  std::vector<fpt*> parameterPointers;
  std::vector<fpt*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 3);
  ASSERT_EQUALS(info2.dimensions[2], 3);

  Vt x(info.outputs());
  x.fill(1.0);
  Vt* y;
  layer.forwardPropagate(&x, y, false);
  for(int i = 0; i < 18; i++)
    ASSERT_EQUALS_DELTA((*y)(i), (fpt) 1.0, (fpt) 1e-5);
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

  Vt gradient = opt.gradient();
  Vt estimatedGradient = opt.gradientFD();
}

void LayerTestCase::multilayerNetwork()
{
  int samples = 10;
  Mt X = Mt::Random(1*6*6, samples);
  Mt Y = Mt::Random(3, samples);
  DirectStorageDataSet ds(X, Y);

  DeepNetwork net;
  net.inputLayer(1, 6, 6);
  net.convolutionalLayer(10, 3, 3, TANH, 0.5);
  net.subsamplingLayer(2, 2, TANH, 0.5);
  net.fullyConnectedLayer(20, TANH, 0.5);
  net.extremeLayer(10, TANH);
  net.outputLayer(3, LINEAR, 0.5);
  net.trainingSet(ds);

  Vt g = net.gradient();
  Vt e = net.gradientFD();
  fpt delta = std::max<fpt>((fpt) 1e-2, 1e-5*e.norm());
  for(int j = 0; j < net.dimension(); j++)
    ASSERT_EQUALS_DELTA(g(j), e(j), (fpt) delta);

  Vt values(samples);
  Mt gradients(samples, net.dimension());
  net.VJ(values, gradients);
  for(int n = 0; n < samples; n++)
  {
    Vt e = net.singleGradientFD(n);
    fpt delta = std::max<fpt>((fpt) 1e-2, 1e-5*e.norm());
    for(int j = 0; j < net.dimension(); j++)
      ASSERT_EQUALS_DELTA(gradients(n, j), e(j), (fpt) delta);
    fpt error = net.error(n);
    ASSERT_EQUALS_DELTA(values(n), error, (fpt) 1e-2);
  }
}
