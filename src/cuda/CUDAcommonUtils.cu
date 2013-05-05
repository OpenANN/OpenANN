#include <CUDAcommonUtils.cuh>
#include <iostream>

bool handleCUDAErrorImpl(cudaError_t result, const char* file, int line)
{
  if(result != cudaSuccess)
  {
    std::cerr << "CUDA error in file " << file << ", line " << line << ":"
        << std::endl << "\t" << cudaGetErrorString(result) << std::endl;
    return false;
  }
  return true;
}

bool handleCUBLASErrorImpl(cublasStatus_t result, const char* file, int line)
{
  if(result != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "CUBLAS error in file " << file << ", line " << line << ":\n"
        << "\t";
    switch(result)
    {
      case CUBLAS_STATUS_NOT_INITIALIZED:
        std::cerr << "the CUDA(tm) Runtime initialization failed";
        break;
      case CUBLAS_STATUS_ALLOC_FAILED:
        std::cerr << "the resources could not be allocated";
        break;
      case CUBLAS_STATUS_INVALID_VALUE:
        std::cerr << "invalid function arguments";
        break;
      case CUBLAS_STATUS_MAPPING_ERROR:
        std::cerr << "there was an error accessing the GPU memory";
        break;
      case CUBLAS_STATUS_EXECUTION_FAILED:
        std::cerr << "the function failed to launch on the GPU";
        break;
      default:
        std::cerr << "unkown error";
    }
    std::cerr << std::endl;
    return false;
  }
  return true;
}
