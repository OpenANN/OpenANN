#ifndef COMPRESSOR_H
#define COMPRESSOR_H

#include <Eigen/Dense>

namespace OpenANN
{

class Compressor
{
  Mt cm;
#ifdef CUDA_AVAILABLE
  float* cmOnDevice;
  float* inputOnDevice;
  float* outputOnDevice;
#endif
public:
  Compressor();
  ~Compressor();
  void reset();
  void init(const Mt& cm);
  Vt compress(const Vt& instance);
};

}

#endif // COMPRESSOR_H
