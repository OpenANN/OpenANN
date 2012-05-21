#ifndef CUDA_COMMON_UTILS
#define CUDA_COMMON_UTILS

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define handleCUDAError(result) \
        handleCUDAErrorImpl(result, __FILE__, __LINE__)

bool handleCUDAErrorImpl(cudaError_t result, const char* file, int line);

#define handleCUBLASError(result) \
        handleCUBLASErrorImpl(result, __FILE__, __LINE__)

bool handleCUBLASErrorImpl(cublasStatus_t result, const char* file, int line);

#endif
