#include "MLPTestCase.h"
#include <MLP.h>
#include <Random.h>

void MLPTestCase::run()
{
  RUN(MLPTestCase, uncompressedBackpropagation);
  RUN(MLPTestCase, compressedBackpropagation);
  RUN(MLPTestCase, uncompressedBackpropagationWithoutBias);
  RUN(MLPTestCase, compressedBackpropagationWithoutBias);
  RUN(MLPTestCase, finishIterationWithoutDataSet);
  RUN(MLPTestCase, network1Backprop);
}

void MLPTestCase::uncompressedBackpropagation()
{
  Mt x(1, 1);
  x.fill(2.0);
  Mt t(1, 1);
  t.fill(5.0);

  OpenANN::MLP mlp;
  mlp
    .input(1)
    .fullyConnectedHiddenLayer(3)
    .fullyConnectedHiddenLayer(3)
    .output(1, OpenANN::MLP::MSE)
    .trainingSet(x, t);

  Mt g = mlp.gradient();
  Vt approximatedGradient = mlp.gradientFD();
  for(int i = 0; i < g.rows(); i++)
  {
    ASSERT_EQUALS_DELTA(g(i), approximatedGradient(i), (fpt) 0.01);
  }
}

void MLPTestCase::compressedBackpropagation()
{
  Mt x(1, 1);
  x.fill(2.0);
  Mt t(1, 1);
  t.fill(5.0);

  OpenANN::MLP mlp;
  mlp
    .input(1)
    .fullyConnectedHiddenLayer(3, OpenANN::MLP::TANH, 4)
    .fullyConnectedHiddenLayer(3, OpenANN::MLP::TANH, 4)
    .output(1, OpenANN::MLP::MSE, OpenANN::MLP::ID, 4)
    .trainingSet(x, t);

  Mt g = mlp.gradient();
  Vt approximatedGradient = mlp.gradientFD();
  for(int i = 0; i < g.rows(); i++)
  {
    ASSERT_EQUALS_DELTA(g(i), approximatedGradient(i), (fpt) 0.01);
  }
}

void MLPTestCase::compressedBackpropagationWithoutBias()
{
  Mt x(1, 1);
  x.fill(2.0);
  Mt t(1, 1);
  t.fill(5.0);

  OpenANN::MLP mlp;
  mlp
    .noBias()
    .input(1)
    .fullyConnectedHiddenLayer(3, OpenANN::MLP::TANH, 3)
    .fullyConnectedHiddenLayer(3, OpenANN::MLP::TANH, 3)
    .output(1, OpenANN::MLP::MSE, OpenANN::MLP::ID, 3)
    .trainingSet(x, t);

  Mt g = mlp.gradient();
  Vt approximatedGradient = mlp.gradientFD();
  for(int i = 0; i < g.rows(); i++)
  {
    ASSERT_EQUALS_DELTA(g(i), approximatedGradient(i), (fpt) 0.01);
  }
}

void MLPTestCase::uncompressedBackpropagationWithoutBias()
{
  Mt x(1, 1);
  x.fill(2.0);
  Mt t(1, 1);
  t.fill(5.0);

  OpenANN::MLP mlp;
  mlp
    .noBias()
    .input(1)
    .fullyConnectedHiddenLayer(3)
    .fullyConnectedHiddenLayer(3)
    .output(1, OpenANN::MLP::MSE)
    .trainingSet(x, t);

  Mt g = mlp.gradient();
  Vt approximatedGradient = mlp.gradientFD();
  for(int i = 0; i < g.rows(); i++)
  {
    ASSERT_EQUALS_DELTA(g(i), approximatedGradient(i), (fpt) 0.01);
  }
}

void MLPTestCase::finishIterationWithoutDataSet()
{
  OpenANN::Logger::deactivate = false;
  OpenANN::MLP mlp;
  mlp.input(1)
     .fullyConnectedHiddenLayer(3)
     .output(1, OpenANN::MLP::MSE);
  ASSERT(!mlp.trainingData);
  ASSERT(!mlp.testData);
  mlp.finishedIteration();
  OpenANN::Logger::deactivate = true;
}

void MLPTestCase::network1Backprop()
{
  int D = 100, F = 10;
  OpenANN::RandomNumberGenerator rng;
  Mt input(D, 1);
  for(int i = 0; i < D; i++)
    input(i, 0) = rng.sampleNormalDistribution<fpt>();
  Mt output(F, 1);
  for(int i = 0; i < F; i++)
    output(i, 0) = rng.generate<fpt>(0.0, 1.0);

  OpenANN::MLP mlp(OpenANN::Logger::NONE, OpenANN::Logger::NONE);
  mlp.input(D)
     .fullyConnectedHiddenLayer(100, OpenANN::MLP::TANH)
     .fullyConnectedHiddenLayer(50, OpenANN::MLP::TANH)
     .output(F, OpenANN::MLP::MSE, OpenANN::MLP::TANH)
     .trainingSet(input, output);
  ASSERT_EQUALS(101*100+101*50+51*10, mlp.dimension());

  Vt g = mlp.gradient(0);
  Vt expectedG = mlp.singleGradientFD(0);
  for(int j = 0; j < g.rows(); j++)
  {
    ASSERT_EQUALS_DELTA(expectedG(j), g(j), (fpt) 1e-4);
  }
}
