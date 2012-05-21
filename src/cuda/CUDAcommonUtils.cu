#include <CUDAcommonUtils.cuh>
#include <iostream>

bool handleCUDAErrorImpl(cudaError_t result, const char* file, int line)
{
  if(result != cudaSuccess)
  {
    std::cerr << "CUDA error in file " << file << ", line " << line << ":\n"
        << "\t";
    switch(result)
    {
      case cudaErrorMissingConfiguration:
        std::cerr << "cudaErrorMissingConfiguration";
        break;
      case cudaErrorMemoryAllocation:
        std::cerr << "cudaErrorMemoryAllocation";
        break;
      case cudaErrorInitializationError:
        std::cerr << "cudaErrorInitializationError";
        break;
      case cudaErrorLaunchFailure:
        std::cerr << "cudaErrorLaunchFailure";
        break;
      case cudaErrorPriorLaunchFailure:
        std::cerr << "cudaErrorPriorLaunchFailure";
        break;
      case cudaErrorLaunchTimeout:
        std::cerr << "cudaErrorLaunchTimeout";
        break;
      case cudaErrorLaunchOutOfResources:
        std::cerr << "cudaErrorLaunchOutOfResources";
        break;
      case cudaErrorInvalidDeviceFunction:
        std::cerr << "cudaErrorInvalidDeviceFunction";
        break;
      case cudaErrorInvalidConfiguration:
        std::cerr << "cudaErrorInvalidConfiguration";
        break;
      case cudaErrorInvalidDevice:
        std::cerr << "cudaErrorInvalidDevice";
        break;
      case cudaErrorInvalidValue:
        std::cerr << "cudaErrorInvalidValue";
        break;
      case cudaErrorInvalidPitchValue:
        std::cerr << "cudaErrorInvalidPitchValue";
        break;
      case cudaErrorInvalidSymbol:
        std::cerr << "cudaErrorInvalidSymbol";
        break;
      case cudaErrorMapBufferObjectFailed:
        std::cerr << "cudaErrorMapBufferObjectFailed";
        break;
      case cudaErrorUnmapBufferObjectFailed:
        std::cerr << "cudaErrorUnmapBufferObjectFailed";
        break;
      case cudaErrorInvalidHostPointer:
        std::cerr << "cudaErrorInvalidHostPointer";
        break;
      case cudaErrorInvalidDevicePointer:
        std::cerr << "cudaErrorInvalidDevicePointer";
        break;
      case cudaErrorInvalidTexture:
        std::cerr << "cudaErrorInvalidTexture";
        break;
      case cudaErrorInvalidTextureBinding:
        std::cerr << "cudaErrorInvalidTextureBinding";
        break;
      case cudaErrorInvalidChannelDescriptor:
        std::cerr << "cudaErrorInvalidChannelDescriptor";
        break;
      case cudaErrorInvalidMemcpyDirection:
        std::cerr << "cudaErrorInvalidMemcpyDirection";
        break;
      case cudaErrorAddressOfConstant:
        std::cerr << "cudaErrorAddressOfConstant";
        break;
      case cudaErrorTextureFetchFailed:
        std::cerr << "cudaErrorTextureFetchFailed";
        break;
      case cudaErrorTextureNotBound:
        std::cerr << "cudaErrorTextureNotBound";
        break;
      case cudaErrorSynchronizationError:
        std::cerr << "cudaErrorSynchronizationError";
        break;
      case cudaErrorInvalidFilterSetting:
        std::cerr << "cudaErrorIvalidFilterSetting";
        break;
      case cudaErrorInvalidNormSetting:
        std::cerr << "cudaErrorInvalidNormSetting";
        break;
      case cudaErrorMixedDeviceExecution:
        std::cerr << "cudaErrorMixedDeviceExecution";
        break;
      case cudaErrorCudartUnloading:
        std::cerr << "cudaErrorCudartUnloading";
        break;
      case cudaErrorUnknown:
        std::cerr << "cudaErrorUnknown";
        break;
      case cudaErrorNotYetImplemented:
        std::cerr << "cudaErrorNotYetImplemented";
        break;
      case cudaErrorMemoryValueTooLarge:
        std::cerr << "cudaErrorMemoryValueTooLarge";
        break;
      case cudaErrorInvalidResourceHandle:
        std::cerr << "cudaErrorInvalidResourceHandle";
        break;
      case cudaErrorNotReady:
        std::cerr << "cudaErrorNotReady";
        break;
      case cudaErrorStartupFailure:
        std::cerr << "cudaErrorStartupFailure";
        break;
      case cudaErrorApiFailureBase:
        std::cerr << "cudaErrorApiFailureBase";
        break;
      default:
        std::cerr << "unkown error";
    }
    std::cerr << std::endl;
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
