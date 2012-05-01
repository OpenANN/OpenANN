#pragma once

#include <CompressionMatrixFactory.h>
#include <io/Logger.h>
#include <Eigen/Dense>
#include <fstream>

namespace OpenANN
{

class FANNFormatLoader
{
public:
  Logger progressLogger;
  int trainingN;
  int testN;
  int D;
  int F;
  Mt trainingInput, trainingOutput, testInput, testOutput;

  int startInput;
  int endInput;

  std::fstream trainingData;
  std::fstream testData;
  int columns;

  bool compress;
  int uncompressedD;
  Mt compressionMatrix;

  FANNFormatLoader(const std::string& benchmark,
      int maxTrainingSamples = -1, int maxTestSamples = -1,
      int startInput = -1, int endInput = -1, bool loadNow = true);
  void prepareLoading(int maxTrainingSamples = -1, int maxTestSamples = -1);
  void allocateMemory();
  void load();
  void load(bool train);
  void compress1D(int paramDim, CompressionMatrixFactory::Transformation transformation);
};

}
