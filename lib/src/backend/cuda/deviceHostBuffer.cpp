// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "deviceHostBuffer.hpp"

#include "deviceStream.hpp"
#include "../common/allocStats.hpp"

#include "gpu/stream.hpp"

#include <cuda_runtime.h>

namespace VideoStitch {
namespace GPU {

// There is no pimpl in the CUDA implementation, so there is no ::get()
// template <typename T>
// const DeviceHostBuffer& HostBuffer<T>::get();

template <typename T>
PotentialValue<HostBuffer<T>> HostBuffer<T>::allocate(size_t numElements, const char* name, unsigned int flags) {
  unsigned int cudaFlags = cudaHostAllocDefault;
  if (flags & GPUHostAllocPinned) {
    cudaFlags |= cudaHostAllocPortable;
  }
  if (flags & GPUHostAllocHostWriteOnly) {
    cudaFlags |= cudaHostAllocWriteCombined;
  }
  T* buf = nullptr;
  size_t byteSize = numElements * sizeof(T);
  if (CUDA_ERROR(cudaHostAlloc((void**)&buf, byteSize, cudaFlags)).ok()) {
    if (buf) {
      hostStats.addPtr(name, (void*)buf, byteSize);
      return PotentialValue<HostBuffer<T>>(HostBuffer(nullptr, buf, numElements));
    }
  }
  return PotentialValue<HostBuffer<T>>(
      {Origin::GPU, ErrType::OutOfResources,
       "Failed to allocate " + std::to_string(numElements * sizeof(T)) +
           " Bytes of host memory. Reduce the project output size and close other applications to free up RAM."});
}

template <typename T>
Status HostBuffer<T>::release() {
  if (hostData) {
    hostStats.deletePtr((void*)hostData);
    return CUDA_ERROR(cudaFreeHost((void*)hostData));
  }
  return Status::OK();
}

template <typename T>
std::size_t HostBuffer<T>::getPoolSize() {
  return hostStats.bytesUsed();
}

template <typename T>
std::vector<std::size_t> HostBuffer<T>::getPoolSizeByDevices() {
  return hostStats.bytesUsedByDevices();
}

template class HostBuffer<unsigned char>;
template class HostBuffer<const unsigned char>;
template class HostBuffer<unsigned int>;
template class HostBuffer<uint16_t>;
template class HostBuffer<int16_t>;
template class HostBuffer<float>;

}  // namespace GPU
}  // namespace VideoStitch
