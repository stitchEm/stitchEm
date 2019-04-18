// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "surface.hpp"
#include "deviceBuffer.hpp"
#include "deviceStream.hpp"
#include "cl_error.hpp"
#include "gpu/allocator.hpp"
#include "gpu/stream.hpp"
#include "gpu/surface.hpp"

#include "../common/allocStats.hpp"

namespace VideoStitch {
namespace GPU {

Surface::~Surface() { delete pimpl; }

CubemapSurface::~CubemapSurface() { delete pimpl; }

bool Surface::operator==(const Surface& other) const {
  if (pimpl && other.pimpl) {
    return *pimpl == *other.pimpl;
  }
  return !pimpl && !other.pimpl;
}

DeviceSurface::~DeviceSurface() {
  if (ownsImage) {
    deviceStats.deletePtr(image);
    clReleaseMemObject(image);
  }
}

DeviceCubemapSurface::~DeviceCubemapSurface() {}

Surface::Surface(DeviceSurface* pimpl, size_t width, size_t height) : pimpl(pimpl), _width(width), _height(height) {}

Surface::Surface(const Surface& other) : pimpl(other.pimpl), _width(other._width), _height(other._height) {}

DeviceSurface& Surface::get() {
  assert(pimpl);
  return *pimpl;
}
const DeviceSurface& Surface::get() const {
  assert(pimpl);
  return *pimpl;
}

}  // namespace GPU

namespace Core {

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-variable"
#elif defined(_MSC_VER)
#pragma warning(disable : 4189)
#endif

Status SourceOpenGLSurface::Pimpl::acquire() {
  acquired = true;
  return CL_ERROR(clEnqueueAcquireGLObjects(stream.get(), 1, &surface->get().raw(), 0, 0, nullptr));
}

Status SourceOpenGLSurface::Pimpl::release() {
  if (acquired) {
    acquired = false;
    return CL_ERROR(clEnqueueReleaseGLObjects(stream.get(), 1, &surface->get().raw(), 0, 0, nullptr));
  }
  return Status::OK();
}

Status PanoOpenGLSurface::Pimpl::acquire() {
  acquired = true;
  cl_mem pbo = buffer.get();
  return CL_ERROR(clEnqueueAcquireGLObjects(stream.get(), 1, &pbo, 0, 0, nullptr));
}

Status PanoOpenGLSurface::Pimpl::release() {
  if (acquired) {
    acquired = false;
    cl_mem pbo = buffer.get();
    return CL_ERROR(clEnqueueReleaseGLObjects(stream.get(), 1, &pbo, 0, 0, nullptr));
  }
  return Status::OK();
}

Status CubemapOpenGLSurface::Pimpl::acquire() {
  for (int i = 0; i < 6; ++i) {
    cl_mem pbo = buffers[i].get();
    FAIL_RETURN(CL_ERROR(clEnqueueAcquireGLObjects(stream.get(), 1, &pbo, 0, 0, nullptr)));
  }
  return Status::OK();
}

Status CubemapOpenGLSurface::Pimpl::release() {
  for (int i = 0; i < 6; ++i) {
    cl_mem pbo = buffers[i].get();
    FAIL_RETURN(CL_ERROR(clEnqueueReleaseGLObjects(stream.get(), 1, &pbo, 0, 0, nullptr)));
  }
  return Status::OK();
}

Potential<PanoOpenGLSurface::Pimpl> PanoOpenGLSurface::Pimpl::create(GPU::Buffer<uint32_t> buffer,
                                                                     GPU::Surface* remapSurf, size_t w, size_t h) {
  PotentialValue<GPU::Stream> stream = GPU::Stream::create();
  if (stream.ok()) {
    return Potential<Pimpl>(new Pimpl(stream.value(), buffer, remapSurf, w, h));
  } else {
    return Potential<Pimpl>(stream.status());
  }
}

Potential<CubemapOpenGLSurface::Pimpl> CubemapOpenGLSurface::Pimpl::create(GPU::Buffer<uint32_t>* buffers,
                                                                           GPU::Buffer<uint32_t> buffer,
                                                                           GPU::CubemapSurface* remapSurf,
                                                                           GPU::Buffer<uint32_t> tmp, size_t w,
                                                                           bool equiangular) {
  PotentialValue<GPU::Stream> stream = GPU::Stream::create();
  if (stream.ok()) {
    return Potential<Pimpl>(new Pimpl(equiangular, buffers, buffer, remapSurf, tmp, w, stream.value()));
  } else {
    return Potential<Pimpl>(stream.status());
  }
}

Potential<SourceOpenGLSurface::Pimpl> SourceOpenGLSurface::Pimpl::create(GPU::Surface* surf) {
  PotentialValue<GPU::Stream> stream = GPU::Stream::create();
  if (stream.ok()) {
    return Potential<Pimpl>(new Pimpl(surf, stream.value()));
  } else {
    return Potential<Pimpl>(stream.status());
  }
}

}  // namespace Core
}  // namespace VideoStitch
