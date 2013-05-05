#include <OpenANN/layers/Compressed.h>
#include <OpenANN/CompressionMatrixFactory.h>
#include <OpenANN/util/Random.h>

namespace OpenANN {

Compressed::Compressed(OutputInfo info, int J, int M, bool bias,
                       ActivationFunction act, const std::string& compression,
                       double stdDev)
  : I(info.outputs()), J(J), M(M), bias(bias), act(act), stdDev(stdDev),
    W(J, I+bias), Wd(J, I+bias), phi(M, I+1), alpha(J, M), alphad(J, M),
    x(0), a(J), y(J), yd(J), deltas(J), e(I+bias)
{
  CompressionMatrixFactory::Transformation transformation =
      CompressionMatrixFactory::SPARSE_RANDOM;
  if(compression == std::string("gaussian"))
    transformation = CompressionMatrixFactory::GAUSSIAN;
  else if(compression == std::string("dct"))
    transformation = CompressionMatrixFactory::DCT;
  else if(compression == std::string("average"))
    transformation = CompressionMatrixFactory::AVERAGE;
  else if(compression == std::string("edge"))
    transformation = CompressionMatrixFactory::EDGE;
  // For compatibility reasons, we create a compression matrix that assumes
  // that there is a bias.
  CompressionMatrixFactory cmf(I+1, M, transformation);
  cmf.createCompressionMatrix(phi);
}

OutputInfo Compressed::initialize(std::vector<double*>& parameterPointers,
                                      std::vector<double*>& parameterDerivativePointers)
{
  parameterPointers.reserve(parameterPointers.size() + J*M);
  parameterDerivativePointers.reserve(parameterDerivativePointers.size() + J*M);
  for(int j = 0; j < J; j++)
  {
    for(int m = 0; m < M; m++)
    {
      parameterPointers.push_back(&alpha(j, m));
      parameterDerivativePointers.push_back(&alphad(j, m));
    }
  }

  initializeParameters();

  OutputInfo info;
  info.dimensions.push_back(J);
  return info;
}

void Compressed::initializeParameters()
{
  RandomNumberGenerator rng;
  for(int j = 0; j < J; j++)
    for(int m = 0; m < M; m++)
      alpha(j, m) = rng.sampleNormalDistribution<double>() * stdDev;
  updatedParameters();
}

void Compressed::updatedParameters()
{
  W = alpha * phi.block(0, 0, M, I+bias);
}

void Compressed::forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout)
{
  this->x = x;
  // Activate neurons
  a = W.leftCols(I) * *x;
  if(bias)
    a += W.rightCols(1);
  // Compute output
  activationFunction(act, a, this->y);
  y = &(this->y);
}

void Compressed::backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout)
{
  // Derive activations
  activationFunctionDerivative(act, y, yd);
  for(int j = 0; j < J; j++)
    deltas(j) = yd(j) * (*ein)(j);
  // Weight derivatives
  Wd.leftCols(I) = deltas * x->transpose();
  if(bias)
    Wd.rightCols(1) = deltas;
  alphad = Wd * phi.block(0, 0, M, I+bias).transpose();
  // Prepare error signals for previous layer
  e = W.leftCols(I).transpose() * deltas;
  eout = &e;
}

Eigen::VectorXd& Compressed::getOutput()
{
  return y;
}

}
