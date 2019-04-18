// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "deviceBuffer.hpp"

#include "../common/allocStats.hpp"
#include "gpu/buffer.hpp"
#include "cuda/error.hpp"

#include <cuda_runtime.h>

namespace VideoStitch {
namespace GPU {

template <typename T>
Buffer<T>::Buffer() : pimpl(nullptr), elements(0) {}

template <typename T>
Buffer<T>::~Buffer() {
  delete pimpl;
}

template <typename T>
Buffer<T>::Buffer(const Buffer& other)
    : pimpl(other.pimpl ? new DeviceBuffer<T>(other.pimpl->data) : nullptr),
      elements(other.elements),
      isSubBuffer(other.isSubBuffer) {}

template <typename T>
Buffer<T>::Buffer(DeviceBuffer<T>* pimpl, size_t num, bool sub)
    : pimpl(pimpl ? new DeviceBuffer<T>(pimpl->data) : nullptr), elements(num), isSubBuffer(sub) {}

template <typename T>
Buffer<T>::Buffer(DeviceBuffer<T>* pimpl, size_t num)
    : pimpl(pimpl ? new DeviceBuffer<T>(pimpl->data) : nullptr), elements(num) {}

template <typename T>
template <typename S>
Buffer<T>::Buffer(DeviceBuffer<S>* pimpl, size_t num)
    : pimpl(pimpl ? new DeviceBuffer<T>(reinterpret_cast<T*>(pimpl->data)) : nullptr), elements(num) {}

template <typename T>
Buffer<T> DeviceBuffer<T>::createBuffer(T* buf, size_t elems) {
  DeviceBuffer<T> tmp(buf);
  return Buffer<T>(&tmp, elems);
}

template <typename T>
PotentialValue<Buffer<T>> Buffer<T>::allocate(size_t numElements, const char* name) {
  void* buf = nullptr;
  size_t byteSize = numElements * sizeof(T);
  if (CUDA_ERROR(cudaMalloc(&buf, byteSize)).ok()) {
    if (buf) {
      deviceStats.addPtr(name, buf, byteSize);
    }
    return PotentialValue<Buffer<T>>(DeviceBuffer<T>::createBuffer((T*)buf, numElements));
  }
  return PotentialValue<Buffer<T>>({Origin::GPU, ErrType::OutOfResources,
                                    "Could not allocate GPU memory. Reduce the project output size and close other "
                                    "applications to free up GPU resources."});
}

template <typename T>
Status Buffer<T>::release() const {
  if (pimpl) {
    if (pimpl->data) {
      deviceStats.deletePtr((void*)pimpl->data);
    }
    const Status releaseStatus = CUDA_ERROR(cudaFree((void*)pimpl->data));
    pimpl->data = nullptr;
    return releaseStatus;
  }
  return Status{Origin::GPU, ErrType::ImplementationError, "Attempting to release an uninitialized buffer"};
}

template <typename T>
Buffer<T> Buffer<T>::createSubBuffer(size_t elementOffset) {
  assert(elementOffset < elements);
  // we only have type info in Buffer, not in DeviceBuffer
  T* typedData = (T*)pimpl->data;
  T* withOffset = typedData + elementOffset;
  DeviceBuffer<T> tmp(withOffset);
  return Buffer<T>(&tmp, elements - elementOffset, true);
}

template <typename T>
const DeviceBuffer<T>& Buffer<T>::get() const {
  assert(pimpl);
  return *pimpl;
}

template <typename T>
DeviceBuffer<T>& Buffer<T>::get() {
  assert(pimpl);
  return *pimpl;
}

template <typename T>
T* Buffer<T>::devicePtr() const {
  assert(pimpl);
  return pimpl->raw();
}

template <typename T>
Buffer<T> Buffer<T>::wrap(T* devicePtr, size_t num) {
  return DeviceBuffer<T>::createBuffer(devicePtr, num);
}

template <typename T>
bool Buffer<T>::wasAllocated() const {
  return pimpl && pimpl->data;
}

template <typename T>
bool Buffer<T>::operator==(const Buffer<T>& other) const {
  if (pimpl && other.pimpl) {
    return *pimpl == *other.pimpl;
  }
  return !pimpl && !other.pimpl;
}

void printBufferPoolStats() { deviceStats.print(std::cout); }

std::size_t getBufferPoolCurrentSize() { return deviceStats.bytesUsed(); }

std::vector<std::size_t> getBufferPoolCurrentSizeByDevices() { return deviceStats.bytesUsedByDevices(); }

// template instantiations
#include "backend/common/deviceBuffer.inst"

}  // namespace GPU
}  // namespace VideoStitch
