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

  void allocateMatrix(float** device, int rows, int cols);
  void freeMatrix(float* device);
  void setMatrix(const float* host, float* device, int rows, int cols);
  void getMatrix(float* host, const float* device, int rows, int cols);
  void multiplyMatrixVector(float* matrix, float* vector, float* result, int rows, int cols);
};

#endif