#include <Compressor.h>
#ifdef CUDA_AVAILABLE
#include <cuBLASInterface.cuh>
#define maxInputDimensionWithoutCuda 10000
#define maxInputDimensionWithCuda 100000
#endif

namespace OpenANN
{

Compressor::Compressor()
#ifdef CUDA_AVAILABLE
  : cmOnDevice(0), inputOnDevice(0), outputOnDevice(0)
#endif
{
}

Compressor::~Compressor()
{
  reset();
}

void Compressor::reset()
{
#ifdef CUDA_AVAILABLE
  if(cmOnDevice)
  {
    CUBLASContext::instance.freeMatrix(cmOnDevice);
    cmOnDevice = 0;
    CUBLASContext::instance.freeMatrix(inputOnDevice);
    inputOnDevice = 0;
    CUBLASContext::instance.freeMatrix(outputOnDevice);
    outputOnDevice = 0;
  }
#endif
}

void Compressor::init(const Mt& cm)
{
  this->cm.resize(cm.rows(), cm.cols());
  this->cm = cm;
#ifdef CUDA_AVAILABLE
  if(cm.cols() > maxInputDimensionWithoutCuda && cm.cols() < maxInputDimensionWithCuda)
  {
    CUBLASContext::instance.allocateMatrix(&cmOnDevice, cm.rows(), cm.cols());
    CUBLASContext::instance.allocateMatrix(&inputOnDevice, cm.cols(), 1);
    CUBLASContext::instance.allocateMatrix(&outputOnDevice, cm.rows(), 1);
    CUBLASContext::instance.setMatrix(cm.data(), cmOnDevice, cm.rows(), cm.cols());
  }
  else
    cmOnDevice = 0;
#endif
}

Vt Compressor::compress(const Vt& instance)
{
#ifdef CUDA_AVAILABLE
  if(instance.rows() > maxInputDimensionWithoutCuda)
  {
    Vt result(cm.rows());
    if(!CUBLASContext::instance.setMatrix(instance.data(), inputOnDevice, instance.rows(), 1))
      abort();
    if(!CUBLASContext::instance.multiplyMatrixVector(cmOnDevice, inputOnDevice, outputOnDevice, cm.rows(), cm.cols()))
      abort();
    if(!CUBLASContext::instance.getMatrix(result.data(), outputOnDevice, result.rows(), 1))
      abort();
    return result;
  }
  else
  {
    return cm * instance;
  }
#else
  return cm * instance;
#endif
}

}
