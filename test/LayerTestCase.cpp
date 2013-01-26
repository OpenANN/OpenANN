#include "LayerTestCase.h"
#include <layers/FullyConnected.h>

using namespace OpenANN;

void LayerTestCase::run()
{
  Logger::deactivate = false; // TODO remove
  RUN(LayerTestCase, fullyConnectedActivate);
}

void LayerTestCase::fullyConnectedActivate()
{
  Logger debugLogger(Logger::CONSOLE);
  FullyConnected layer(3, 2, true, TANH, 0.05);
  std::list<fpt*> parameterPointers;
  std::list<fpt*> parameterDerivativePointers;
  layer.initialize(parameterPointers, parameterDerivativePointers);
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

  layer.accumulate(&e);
  layer.gradient();
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

  Vt* e2;
  layer.backpropagate(e2);
  ASSERT(e2 != 0);
}
