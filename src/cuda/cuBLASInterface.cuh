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

  bool allocateMatrix(double** device, int rows, int cols);
  bool freeMatrix(double* device);
  bool setMatrix(const double* host, double* device, int rows, int cols);
  bool getMatrix(double* host, const double* device, int rows, int cols);
  bool multiplyMatrixVector(double* matrix, double* vector, double* result, int rows, int cols);
};

#endif