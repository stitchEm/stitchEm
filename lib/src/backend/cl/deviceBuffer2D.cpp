// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "deviceBuffer2D.hpp"

#include "context.hpp"

#include "gpu/2dBuffer.hpp"
#include "../common/allocStats.hpp"

namespace VideoStitch {
namespace GPU {

/**
 * Compute an optimal pitch (= distance in byte between rows in a 2D array) such that each new row begins at an address
 * that is optimal for coalescing, and then allocates a memory area as large as pitch times the number of rows you
 * specify.
 *
 * The optimal pitch is computed by
 * (1) getting the base address alignment preference for your card(CL_DEVICE_MEM_BASE_ADDR_ALIGN property
 * with clGetDeviceInfo : note that the returned value is in bits); let's call this base
 * (2) find the largest multiple of base that is no less than your natural data pitch (sizeof(type) times number of
 * columns); this will be the pitch. Then allocate pitch times number of rows bytes, and pass the pitch information to
 * kernels.
 */
PotentialValue<Buffer2D> Buffer2D::allocate(size_t width, size_t height, const char* name) {
  const auto& potContext = getContext();
  FAIL_RETURN(potContext.status());
  const auto& ctx = potContext.value();
  cl_int base;
  cl_int err = clGetDeviceInfo(ctx.deviceID(), CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(base), &base, nullptr);
  if (err != CL_SUCCESS) {
    assert(false);
  }
  assert(base % 8 == 0);
  size_t mod = width % (base / 8);
  size_t pitch = width;
  if (mod > 0) {
    pitch = width + (base / 8) - mod;
  }

  cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, pitch * height, nullptr, &err);
  Status status = CL_ERROR(err);
  if (status.ok()) {
    deviceStats.addPtr(name, buf, pitch * height);
    return PotentialValue<Buffer2D>(Buffer2D(new DeviceBuffer2D(buf), width, height, pitch));
  } else {
    return PotentialValue<Buffer2D>({Origin::GPU, ErrType::OutOfResources,
                                     "Could not allocate GPU memory. Reduce the project output size and close other "
                                     "applications to free up GPU memory."});
  }
}

Status Buffer2D::release() const {
  if (pimpl && pimpl->image) {
    deviceStats.deletePtr(pimpl->image);
    return CL_ERROR(clReleaseMemObject(pimpl->image));
  }
  return Status::OK();
}

Buffer2D::~Buffer2D() { delete pimpl; }

Buffer2D::Buffer2D(const Buffer2D& other)
    : pimpl(other.pimpl ? new DeviceBuffer2D(other.pimpl->image) : nullptr),
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
