#include "cuBLASInterface.cuh"
#include <CUDAcommonUtils.cuh>
#include <cmath>

CUBLASContext CUBLASContext::instance;

CUBLASContext::CUBLASContext()
{
  handleCUBLASError(cublasCreate(&handle));
}

CUBLASContext::~CUBLASContext()
{
  handleCUBLASError(cublasDestroy(handle));
}

void CUBLASContext::allocateMatrix(float** device, int rows, int cols)
{
  const int size = rows*cols * sizeof(float);
  handleCUDAError(cudaMalloc((void**)device, size));
}

void CUBLASContext::freeMatrix(float* device)
{
  handleCUDAError(cudaFree(device));
}

void CUBLASContext::setMatrix(const float* host, float* device, int rows, int cols)
{
  if(cols == 1)
  {
    handleCUBLASError(cublasSetVector(rows, sizeof(*host), host, 1, device, 1));
  }
  else
  {
    handleCUBLASError(cublasSetMatrix(rows, cols, sizeof(*host), host, rows, device, rows));
  }
}

void CUBLASContext::getMatrix(float* host, const float* device, int rows, int cols)
{
  if(cols == 1)
  {
    handleCUBLASError(cublasGetVector(rows, sizeof(*device), device, 1, host, 1));
  }
  else
  {
    handleCUBLASError(cublasGetMatrix(rows, cols, sizeof(*device), device, rows, host, rows));
  }
}

void CUBLASContext::multiplyMatrixVector(float* matrix, float* vector, float* result, int rows, int cols)
{
  float alpha = 1.0, beta = 0.0;
  handleCUBLASError(cublasSgemv(handle, CUBLAS_OP_N, rows, cols, &alpha, matrix, rows, vector, 1, &beta, result, 1));
}
