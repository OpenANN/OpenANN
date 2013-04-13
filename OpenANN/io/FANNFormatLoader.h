#pragma once

#include <OpenANN/io/Logger.h>
#include <Eigen/Dense>
#include <fstream>

namespace OpenANN
{

/**
 * @class FANNFormatLoader
 *
 * Load training and test sets from Fast Artificial Neural Networks library.
 */
class FANNFormatLoader
{
public:
  Logger progressLogger;
  int trainingN;
  int testN;
  int D;
  int F;
  Mt trainingInput, trainingOutput, testInput, testOutput;

  std::fstream trainingData;
  std::fstream testData;

  /**
   * Create a FANNFormatLoader.
   * @param benchmark name of the benchmark
   * @param maxTrainingSamples maximum number of training examples
   * @param maxTestSamples maximum number of test examples
   * @param loadNow load data sets during creation
   */
  FANNFormatLoader(const std::string& benchmark, int maxTrainingSamples = -1,
                   int maxTestSamples = -1, bool loadNow = true);
  void load();
private:
  void prepareLoading(int maxTrainingSamples = -1, int maxTestSamples = -1);
  void allocateMemory();
  void load(bool train);
};

}
