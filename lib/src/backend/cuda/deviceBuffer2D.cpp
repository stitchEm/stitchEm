// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "deviceBuffer2D.hpp"

#include "../common/allocStats.hpp"

#include "cuda/error.hpp"
#include "gpu/2dBuffer.hpp"

namespace VideoStitch {
namespace GPU {

PotentialValue<Buffer2D> Buffer2D::allocate(size_t width, size_t height, const char* name) {
  unsigned char* buf = nullptr;
  size_t pitch;
  if (CUDA_ERROR(cudaMallocPitch((void**)&buf, &pitch, width, height)).ok()) {
    if (buf) {
      deviceStats.addPtr(name, buf, pitch * height);
      return PotentialValue<Buffer2D>(Buffer2D(new DeviceBuffer2D(buf), width, height, pitch));
    }
  }
  return PotentialValue<Buffer2D>({Origin::GPU, ErrType::OutOfResources,
                                   "Could not allocate GPU memory. Reduce the project output size and close other "
                                   "applications to free up GPU memory."});
}

Status Buffer2D::release() const {
  if (pimpl && pimpl->data) {
    deviceStats.deletePtr((void*)pimpl->data);
    const Status releaseStatus = CUDA_ERROR(cudaFree((void*)pimpl->data));
    return releaseStatus;
  }
  return Status::OK();
}

Buffer2D::~Buffer2D() { delete pimpl; }

Buffer2D::Buffer2D(const Buffer2D& other)
    : pimpl(other.pimpl ? new DeviceBuffer2D(other.pimpl->data) : nullptr),
      width(other.width),
      height(other.height),
      pitch(other.pitch) {}

bool Buffer2D::operator==(const Buffer2D& other) const {
  if (pimpl && other.pimpl) {
    return *pimpl == *other.pimpl;
  }
  return !pimpl && !other.pimpl;
}

void* Buffer2D::devicePtr() const {
  if (pimpl) {
    return pimpl->raw();
  }
  return nullptr;
}

}  // namespace GPU
}  // namespace VideoStitch
