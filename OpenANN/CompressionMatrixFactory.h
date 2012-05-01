#pragma once

#include <Eigen/Dense>
#include <vector>

namespace OpenANN
{

class CompressionMatrixFactory
{
public:
  bool compress;
  enum Transformation {
    DCT, GAUSSIAN, AVERAGE, EDGE
  } transformation;
  int inputDim;
  int paramDim;

  CompressionMatrixFactory();
  CompressionMatrixFactory(int inputDim, int paramDim,
      Transformation transformation = DCT);
  void createCompressionMatrix(Mt& cm);

private:
  void fillCompressionMatrix(Mt& cm);
};

}
