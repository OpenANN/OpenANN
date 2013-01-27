#include "LayerTestCase.h"
#include <layers/FullyConnected.h>
#include <layers/Convolutional.h>

using namespace OpenANN;

void LayerTestCase::run()
{
  Logger::deactivate = false; // TODO remove
  RUN(LayerTestCase, fullyConnected);
  RUN(LayerTestCase, convolutional);
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
