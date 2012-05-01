#pragma once

#include <Eigen/Dense>
#include <vector>

namespace OpenANN
{

class CompressionMatrixFactory
{
public:
  bool compress;
  enum Dimension {
    ONE, TWO, THREE
  } dimension;
  enum Transformation {
    DCT, GAUSSIAN, AVERAGE, EDGE
  } transformation;
  int firstInputDim;
  int secondInputDim;
  int firstParamDim;
  int secondParamDim;
  int thirdParamDim;
  std::vector<std::vector<fpt> >* t;

  CompressionMatrixFactory();
  CompressionMatrixFactory(int firstInputDim, int firstParamDim,
                           Transformation transformation = DCT);
  CompressionMatrixFactory(int firstInputDim, int secondInputDim,
                           int firstParamDim, int secondParamDim,
                           Transformation transformation = DCT);
  CompressionMatrixFactory(int firstInputDim, int secondInputDim, std::vector<std::vector<fpt> >* t,
                           int firstParamDim, int secondParamDim, int thirdParamDim,
                           Transformation transformation = DCT);
  void createCompressionMatrix(Mt& orthogonalFunctionsMatrix);

private:
  void create1DCompressionMatrix(Mt& orthogonalFunctionsMatrix);
  void create2DCompressionMatrix(Mt& orthogonalFunctionsMatrix);
  void create3DCompressionMatrix(Mt& orthogonalFunctionsMatrix);
};

}
