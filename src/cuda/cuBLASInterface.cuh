#ifndef MATRIX_VECTOR_MULTIPLICATION
#define MATRIX_VECTOR_MULTIPLICATION

#include <cuda_runtime.h>
#include <cublas_v2.h>

class CUBLASContext
{
  CUBLASContext();
  ~CUBLASContext();
public:
  static CUBLASContext instance;
  cublasHandle_t handle;

  bool allocateMatrix(float** device, int rows, int cols);
  bool freeMatrix(float* device);
  bool setMatrix(const float* host, float* device, int rows, int cols);
  bool getMatrix(float* host, const float* device, int rows, int cols);
  bool multiplyMatrixVector(float* matrix, float* vector, float* result, int rows, int cols);
};

#endif