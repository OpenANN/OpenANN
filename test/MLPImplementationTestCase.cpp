#include "MLPImplementationTestCase.h"
#include <MLPImplementation.h>
#include <Random.h>

using namespace OpenANN;

void MLPImplementationTestCase::run()
{
  RUN(MLPImplementationTestCase, twoDimensionalOrthogonalFunctionsMatrix);
  RUN(MLPImplementationTestCase, uncompressedForwardPropagation);
  RUN(MLPImplementationTestCase, compressedForwardPropagation);
}

void MLPImplementationTestCase::twoDimensionalOrthogonalFunctionsMatrix()
{
  MLPImplementation fmlp;
  fmlp.parametersX = 3;
  fmlp.parametersY = 3;
  MLPImplementation::LayerInfo layerInfo1(MLPImplementation::LayerInfo::INPUT, 2, 9, MLPImplementation::ID);
  layerInfo1.nodesPerDimension.push_back(3);
  layerInfo1.nodesPerDimension.push_back(3);
  fmlp.layerInfos.push_back(layerInfo1);
  MLPImplementation::LayerInfo layerInfo2(MLPImplementation::LayerInfo::FULLY_CONNECTED, 1, 1, MLPImplementation::ID);
  layerInfo2.compress(9);
  fmlp.layerInfos.push_back(layerInfo2);
  MLPImplementation::LayerInfo layerInfo3(MLPImplementation::LayerInfo::OUTPUT, 1, 9, MLPImplementation::ID);
  fmlp.layerInfos.push_back(layerInfo3);
  fmlp.init();

  Mt expectedOrthogonalFunctionsMatrix(9, 9);
  expectedOrthogonalFunctionsMatrix <<
      1,  1,  1,  1,  1,  1,  1,  1,  1,
      1,  0, -1,  1,  0, -1,  1,  0, -1,
      1, -1,  1,  1, -1,  1,  1, -1,  1,
      1,  1,  1,  0,  0,  0, -1, -1, -1,
      1,  0, -1,  0,  0,  0, -1,  0,  1,
      1, -1,  1,  0,  0,  0, -1,  1, -1,
      1,  1,  1, -1, -1, -1,  1,  1,  1,
      1,  0, -1, -1,  0,  1,  1,  0, -1,
      1, -1,  1, -1,  1, -1,  1, -1,  1;

  for(int m = 0; m < 9; m++)
  {
    for(int i = 0; i < 9; i++)
    {
      ASSERT_EQUALS_DELTA(expectedOrthogonalFunctionsMatrix(m, i), fmlp.orthogonalFunctions[0](m, i), (fpt) 1e-10);
    }
  }
}

void MLPImplementationTestCase::uncompressedForwardPropagation()
{
  MLPImplementation fmlp;
  MLPImplementation::LayerInfo layerInfo1(MLPImplementation::LayerInfo::INPUT, 1, 1, MLPImplementation::ID);
  fmlp.layerInfos.push_back(layerInfo1);
  MLPImplementation::LayerInfo layerInfo2(MLPImplementation::LayerInfo::FULLY_CONNECTED, 1, 1, MLPImplementation::TANH);
  fmlp.layerInfos.push_back(layerInfo2);
  MLPImplementation::LayerInfo layerInfo3(MLPImplementation::LayerInfo::OUTPUT, 1, 1, MLPImplementation::ID);
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
  MLPImplementation fmlp;
  MLPImplementation::LayerInfo layerInfo1(MLPImplementation::LayerInfo::INPUT, 1, 1, MLPImplementation::ID);
  fmlp.layerInfos.push_back(layerInfo1);
  MLPImplementation::LayerInfo layerInfo2(MLPImplementation::LayerInfo::FULLY_CONNECTED, 1, 1, MLPImplementation::TANH);
  layerInfo2.compress(2);
  fmlp.layerInfos.push_back(layerInfo2);
  MLPImplementation::LayerInfo layerInfo3(MLPImplementation::LayerInfo::OUTPUT, 1, 1, MLPImplementation::ID);
  layerInfo3.compress(2);
  fmlp.layerInfos.push_back(layerInfo3);
  fmlp.init();
  fmlp.constantParameters(1.0);

  Vt x(1, 1);
  x.fill(1.0);
  Vt y = fmlp(x);
  ASSERT_EQUALS_DELTA(y(0, 0), (fpt) 1.92805516, (fpt) 1e-5);
}
