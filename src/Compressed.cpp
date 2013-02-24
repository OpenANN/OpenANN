#include <layers/Compressed.h>
#include <CompressionMatrixFactory.h>
#include <Random.h>

namespace OpenANN {

Compressed::Compressed(OutputInfo info, int J, int M, bool bias,
                       ActivationFunction act, const std::string& compression,
                       fpt stdDev, fpt dropoutProbability)
  : I(info.outputs()), J(J), M(M), bias(bias), act(act), stdDev(stdDev),
    dropoutProbability(dropoutProbability), W(J, I), Wd(J, I), phi(M, I),
    alpha(J, M), alphad(J, M), x(0), a(J), y(J+bias), yd(J), deltas(J), e(I)
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
  CompressionMatrixFactory cmf(I, M, transformation);
  cmf.createCompressionMatrix(phi);
}

OutputInfo Compressed::initialize(std::vector<fpt*>& parameterPointers,
                                      std::vector<fpt*>& parameterDerivativePointers)
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

  // Bias component will not change after initialization
  if(bias)
    y(J) = fpt(1.0);

  initializeParameters();

  OutputInfo info;
  info.bias = bias;
  info.dimensions.push_back(J);
  return info;
}

void Compressed::initializeParameters()
{
  RandomNumberGenerator rng;
  for(int j = 0; j < J; j++)
    for(int m = 0; m < M; m++)
      alpha(j, m) = rng.sampleNormalDistribution<fpt>() * stdDev;
  updatedParameters();
}

void Compressed::updatedParameters()
{
  W = alpha * phi;
}

void Compressed::forwardPropagate(Vt* x, Vt*& y, bool dropout)
{
  this->x = x;
  // Activate neurons
  a = W * *x;
  // Compute output
  activationFunction(act, a, this->y);
  if(dropout)
  {
    RandomNumberGenerator rng;
    for(int j = 0; j < J; j++)
      if(rng.generate<fpt>(0.0, 1.0) < dropoutProbability)
        this->y(j) = (fpt) 0;
  }
  else if(dropoutProbability > 0.0)
  {
    // Hinton, 2012: "At test time, we use the "mean network" [...] to
    // compensate for the fact that [all] of them are active."
    // Scaling the outputs is equivalent to scaling the outgoing weights.
    this->y *= (1.0 - dropoutProbability);
    if(bias)
      this->y(J) = 1.0;
  }
  y = &(this->y);
}

void Compressed::backpropagate(Vt* ein, Vt*& eout)
{
  // Derive activations
  activationFunctionDerivative(act, y, yd);
  for(int j = 0; j < J; j++)
    deltas(j) = yd(j) * (*ein)(j);
  // Weight derivatives
  Wd = deltas * x->transpose();
  alphad = Wd * phi.transpose();
  // Prepare error signals for previous layer
  e = W.transpose() * deltas;
  eout = &e;
}

Vt& Compressed::getOutput()
{
  return y;
}

}
