#pragma once

#include <CompressionMatrixFactory.h>
#include <io/Logger.h>

namespace OpenANN
{

class Compressor
{
  Logger debugLogger;
  Mt cm;
#ifdef CUDA_AVAILABLE
  float* cmOnDevice;
  float* inputOnDevice;
  float* outputOnDevice;
#endif
public:
  Compressor();
  ~Compressor();
  void init(const Mt& cm);
  Vt compress(const Vt& instance);
};

}
