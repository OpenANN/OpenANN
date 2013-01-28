#include "LayerTestCase.h"
#include <layers/FullyConnected.h>
#include <layers/Convolutional.h>
#include <Optimizable.h>

using namespace OpenANN;

class LayerOptimizable : public Optimizable
{
  Layer& layer;
  std::list<fpt*> parameters;
  std::list<fpt*> derivatives;
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
    std::list<fpt*>::const_iterator it = parameters.begin();
    for(int i = 0; i < dimension(); i++, it++)
      params(i) = **it;
    return params;
  }

  virtual void setParameters(const Vt& parameters)
  {
    std::list<fpt*>::const_iterator it = this->parameters.begin();
    for(int i = 0; i < dimension(); i++, it++)
      **it = parameters(i);
  }

  virtual fpt error()
  {
    Vt* output;
    layer.forwardPropagate(&input, output);
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
    layer.forwardPropagate(&input, output);
    Vt diff = *output;
    for(int i = 0; i < desired.rows(); i++)
      diff(i) = (*output)(i) - desired(i);
    Vt* e;
    layer.backpropagate(&diff, e);
    Vt derivs(dimension());
    std::list<fpt*>::const_iterator it = derivatives.begin();
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
  Logger::deactivate = false; // TODO remove
  RUN(LayerTestCase, fullyConnected);
  RUN(LayerTestCase, fullyConnectedGradient);
  RUN(LayerTestCase, convolutional);
  RUN(LayerTestCase, convolutionalGradient);
}

void LayerTestCase::fullyConnected()
{
  Logger debugLogger(Logger::CONSOLE);

  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(3);
  FullyConnected layer(info, 2, true, TANH, 0.05);

  std::list<fpt*> parameterPointers;
  std::list<fpt*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT(info2.bias);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(*(info2.dimensions.begin()), 3);

  for(std::list<fpt*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 1.0;
  Vt x(3);
  x << 0.5, 1.0, 2.0;
  Vt e(3);
  e << 1.0, 2.0, 0.0;

  Vt* y;
  layer.forwardPropagate(&x, y);
  ASSERT(y != 0);
  ASSERT_EQUALS_DELTA((*y)(0), (fpt) tanh(3.5), (fpt) 1e-10);
  ASSERT_EQUALS_DELTA((*y)(1), (fpt) tanh(3.5), (fpt) 1e-10);
  ASSERT_EQUALS_DELTA((*y)(2), (fpt) 1.0, (fpt) 1e-10);

  Vt* e2;
  layer.backpropagate(&e, e2);
  Vt Wd(6);
  int i = 0;
  for(std::list<fpt*>::iterator it = parameterDerivativePointers.begin();
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
  FullyConnected layer(info, 2, true, TANH, 0.05);
  LayerOptimizable opt(layer, info);

  Vt gradient = opt.gradient();
  Vt estimatedGradient = opt.gradientFD();
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), (fpt) 1e-4);
}

void LayerTestCase::convolutional()
{
  Logger debugLogger(Logger::CONSOLE);

  OutputInfo info;
  info.bias = false;
  info.dimensions.push_back(2);
  info.dimensions.push_back(4);
  info.dimensions.push_back(4);
  Convolutional layer(info, 2, 3, 3, true, TANH, 0.05);
  std::list<fpt*> parameterPointers;
  std::list<fpt*> parameterDerivativePointers;
  OutputInfo info2 = layer.initialize(parameterPointers, parameterDerivativePointers);
  ASSERT_EQUALS(info2.dimensions.size(), 3);
  ASSERT_EQUALS(info2.dimensions[0], 2);
  ASSERT_EQUALS(info2.dimensions[1], 2);
  ASSERT_EQUALS(info2.dimensions[2], 2);

  for(std::list<fpt*>::iterator it = parameterPointers.begin();
      it != parameterPointers.end(); it++)
    **it = 0.01;

  Vt x(info.outputs());
  x.fill(1.0);
  Vt* y;
  layer.forwardPropagate(&x, y);
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
  info.dimensions.push_back(2);
  info.dimensions.push_back(4);
  info.dimensions.push_back(4);
  Convolutional layer(info, 2, 3, 3, true, TANH, 0.05);
  LayerOptimizable opt(layer, info);

  Vt gradient = opt.gradient();
  Vt estimatedGradient = opt.gradientFD();
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), (fpt) 1e-4);
}
