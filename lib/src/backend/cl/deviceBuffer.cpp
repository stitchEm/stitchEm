// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "deviceBuffer.hpp"

#include "gpu/vectorTypes.hpp"

#include "context.hpp"

#include "../common/allocStats.hpp"

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
    : pimpl(other.pimpl ? new DeviceBuffer<T>(other.pimpl) : nullptr),
      elements(other.elements),
      isSubBuffer(other.isSubBuffer) {}

template <typename T>
Buffer<T>::Buffer(DeviceBuffer<T>* pimpl, size_t num, bool sub)
    : pimpl(pimpl ? new DeviceBuffer<T>(pimpl) : nullptr), elements(num), isSubBuffer(sub) {}

template <typename T>
Buffer<T>::Buffer(DeviceBuffer<T>* pimpl, size_t num)
    : pimpl(pimpl ? new DeviceBuffer<T>(pimpl) : nullptr), elements(num) {}

template <typename T>
template <typename S>
Buffer<T>::Buffer(DeviceBuffer<S>* pimpl, size_t num)
    : pimpl(pimpl ? new DeviceBuffer<T>(pimpl) : nullptr), elements(num) {}

template <typename T>
Buffer<T> DeviceBuffer<T>::createBuffer(cl_mem buf, size_t elems) {
  DeviceBuffer tmp(buf);
  return Buffer<T>(&tmp, elems);
}

template <typename T>
PotentialValue<Buffer<T>> Buffer<T>::allocate(size_t numElements, const char* name) {
  const auto& potContext = getContext();
  FAIL_RETURN(potContext.status());
  const auto& ctx = potContext.value();
  int err;
  size_t byteSize = numElements * sizeof(T);
  cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, byteSize, nullptr, &err);
  Status status = CL_ERROR(err);
  if (status.ok()) {
    deviceStats.addPtr(name, buf, byteSize);
    return PotentialValue<Buffer<T>>(DeviceBuffer<T>::createBuffer(buf, numElements));
  } else {
    return status;
  }
}

template <typename T>
Status Buffer<T>::release() const {
  if (pimpl && pimpl->data) {
    deviceStats.deletePtr(pimpl->data);
    return CL_ERROR(clReleaseMemObject(pimpl->data));
  }
  return Status{Origin::GPU, ErrType::ImplementationError, "Attempting to release an uninitialized buffer"};
}

template <typename T>
Buffer<T> Buffer<T>::createSubBuffer(size_t elementOffset) {
  assert(elementOffset < elements);
  assert(!isSubBuffer);
  cl_buffer_region region = {0};
  region.origin = elementOffset * sizeof(T);
  region.size = (elements - elementOffset) * sizeof(T);
  cl_int err;
  cl_mem sub = clCreateSubBuffer(pimpl->data, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
  assert(CL_ERROR(err).ok());
  DeviceBuffer<T> tmp(sub);
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
  return (T*)pimpl->raw();
}

template <typename T>
Buffer<T> Buffer<T>::wrap(T* devicePtr, size_t num) {
  return DeviceBuffer<T>::createBuffer((cl_mem)devicePtr, num);
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

// template instantiations
#include "backend/common/deviceBuffer.inst"

}  // namespace GPU
}  // namespace VideoStitch
