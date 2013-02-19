#include "IntrinsicPlasticityTestCase.h"
#include <IntrinsicPlasticity.h>
#include <Random.h>
#include <optimization/MBSGD.h>

void IntrinsicPlasticityTestCase::run()
{
  RUN(IntrinsicPlasticityTestCase, learn);
}

void IntrinsicPlasticityTestCase::learn()
{
  OpenANN::IntrinsicPlasticity ip(2, 0.2);

  OpenANN::RandomNumberGenerator rng;
  int samples = 10000;
  Mt X(2, samples);
  Mt Y(2, samples);
  for(int i = 0; i < samples; i++)
  {
    for(int j = 0; j < 2; j++)
    {
      X(j, i) = rng.sampleNormalDistribution<fpt>();
      Y(j, i) = rng.sampleNormalDistribution<fpt>();
    }
  }

  ip.initialize();
  Vt p = ip.currentParameters();
  ASSERT_EQUALS_DELTA((fpt) p(0), (fpt) 1.0, (fpt) 1e-3);
  ASSERT_EQUALS_DELTA((fpt) p(1), (fpt) 1.0, (fpt) 1e-3);
  ASSERT_NOT_EQUALS((fpt) p(2), (fpt) 0.0);
  ASSERT_NOT_EQUALS((fpt) p(3), (fpt) 0.0);
  p(2) = 1e-3;
  p(3) = 1e-3;
  ip.setParameters(p);
  p = ip.currentParameters();
  ASSERT_EQUALS_DELTA((fpt) p(0), (fpt) 1.0, (fpt) 1e-3);
  ASSERT_EQUALS_DELTA((fpt) p(1), (fpt) 1.0, (fpt) 1e-3);
  ASSERT_NOT_EQUALS((fpt) p(2), (fpt) 0.0);
  ASSERT_NOT_EQUALS((fpt) p(3), (fpt) 0.0);

  Vt y(2);
  y.fill(0.0);
  for(int i = 0; i < samples; i++)
    y += ip(X.col(i));
  Vt mean = y / (fpt) samples;
  ASSERT_EQUALS_DELTA((fpt) mean(0), (fpt) 0.5, (fpt) 1e-2);
  ASSERT_EQUALS_DELTA((fpt) mean(1), (fpt) 0.5, (fpt) 1e-2);

  ip.trainingSet(X, Y);
  OpenANN::MBSGD sgd(2e-5, 1.0, 2e-5, 0.0, 0.0, 0.0, 1);
  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 50;
  sgd.setOptimizable(ip);
  sgd.setStopCriteria(stop);
  while(sgd.step());

  y.fill(0.0);
  for(int i = 0; i < samples; i++)
    y += ip(X.col(i));
  mean = y / (fpt) samples;
  ASSERT_EQUALS_DELTA((fpt) mean(0), (fpt) 0.2, (fpt) 1e-2);
  ASSERT_EQUALS_DELTA((fpt) mean(1), (fpt) 0.2, (fpt) 1e-2);
}
