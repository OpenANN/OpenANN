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

bool CUBLASContext::allocateMatrix(double** device, int rows, int cols)
{
  const int size = rows*cols * sizeof(double);
  return handleCUDAError(cudaMalloc((void**)device, size));
}

bool CUBLASContext::freeMatrix(double* device)
{
  return handleCUDAError(cudaFree(device));
}

bool CUBLASContext::setMatrix(const double* host, double* device, int rows, int cols)
{
  if(cols == 1)
  {
    return handleCUBLASError(cublasSetVector(rows, sizeof(*host), host, 1, device, 1));
  }
  else
  {
    return handleCUBLASError(cublasSetMatrix(rows, cols, sizeof(*host), host, rows, device, rows));
  }
}

bool CUBLASContext::getMatrix(double* host, const double* device, int rows, int cols)
{
  if(cols == 1)
  {
    return handleCUBLASError(cublasGetVector(rows, sizeof(*device), device, 1, host, 1));
  }
  else
  {
    return handleCUBLASError(cublasGetMatrix(rows, cols, sizeof(*device), device, rows, host, rows));
  }
}

bool CUBLASContext::multiplyMatrixVector(double* matrix, double* vector, double* result, int rows, int cols)
{
  double alpha = 1.0, beta = 0.0;
  return handleCUBLASError(cublasDgemv(handle, CUBLAS_OP_N, rows, cols, &alpha, matrix, rows, vector, 1, &beta, result, 1));
}
