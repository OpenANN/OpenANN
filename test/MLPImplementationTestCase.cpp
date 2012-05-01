#include "MLPImplementationTestCase.h"
#include <MLPImplementation.h>
#include <Random.h>

void MLPImplementationTestCase::run()
{
  RUN(MLPImplementationTestCase, uncompressedForwardPropagation);
  RUN(MLPImplementationTestCase, compressedForwardPropagation);
}

void MLPImplementationTestCase::uncompressedForwardPropagation()
{
  OpenANN::MLPImplementation fmlp;
  OpenANN::MLPImplementation::LayerInfo layerInfo1(OpenANN::MLPImplementation::LayerInfo::INPUT, 1, 1, OpenANN::MLPImplementation::ID);
  fmlp.layerInfos.push_back(layerInfo1);
  OpenANN::MLPImplementation::LayerInfo layerInfo2(OpenANN::MLPImplementation::LayerInfo::FULLY_CONNECTED, 1, 1, OpenANN::MLPImplementation::TANH);
  fmlp.layerInfos.push_back(layerInfo2);
  OpenANN::MLPImplementation::LayerInfo layerInfo3(OpenANN::MLPImplementation::LayerInfo::OUTPUT, 1, 1, OpenANN::MLPImplementation::ID);
  fmlp.layerInfos.push_back(layerInfo3);
  fmlp.init();
  fmlp.constantParameters(1.0);

  Vt x(1, 1);
  x.fill(1.0);
  Vt y = fmlp(x);
  ASSERT_EQUALS_DELTA(y(0, 0), (fpt) 1.96402758, (fpt) 1e-5);
}

void MLPImplementationTestCase::compressedForwardPropagation()
{
  OpenANN::MLPImplementation fmlp;
  OpenANN::MLPImplementation::LayerInfo layerInfo1(OpenANN::MLPImplementation::LayerInfo::INPUT, 1, 1, OpenANN::MLPImplementation::ID);
  fmlp.layerInfos.push_back(layerInfo1);
  OpenANN::MLPImplementation::LayerInfo layerInfo2(OpenANN::MLPImplementation::LayerInfo::FULLY_CONNECTED, 1, 1, OpenANN::MLPImplementation::TANH);
  layerInfo2.compress(2);
  fmlp.layerInfos.push_back(layerInfo2);
  OpenANN::MLPImplementation::LayerInfo layerInfo3(OpenANN::MLPImplementation::LayerInfo::OUTPUT, 1, 1, OpenANN::MLPImplementation::ID);
  layerInfo3.compress(2);
  fmlp.layerInfos.push_back(layerInfo3);
  fmlp.init();
  fmlp.constantParameters(1.0);

  Vt x(1, 1);
  x.fill(1.0);
  Vt y = fmlp(x);
  ASSERT_EQUALS_DELTA(y(0, 0), (fpt) 1.92805516, (fpt) 1e-5);
}
