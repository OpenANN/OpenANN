#include "DropoutTestCase.h"
#include "FiniteDifferences.h"
#include "LayerAdapter.h"
#include <OpenANN/layers/Dropout.h>

using namespace OpenANN;

void DropoutTestCase::run()
{
  RUN(DropoutTestCase, dropout);
}

void DropoutTestCase::dropout()
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
  Eigen::MatrixXd x(2, samples);
  x.fill(1.0);
  Eigen::MatrixXd* y;
  layer.forwardPropagate(&x, y, true);
  double mean = y->sum() / samples;
  ASSERT_EQUALS_DELTA(mean/2.0, 0.5, 0.01);
  // After training, the output should be scaled down
  layer.forwardPropagate(&x, y, false);
  mean = y->sum() / samples;
  ASSERT_EQUALS(mean/2.0, 0.5);
}
