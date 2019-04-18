// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "surface.hpp"
#include "deviceStream.hpp"

#include "cuda/error.hpp"

#include "../common/allocStats.hpp"

namespace VideoStitch {
namespace GPU {

Surface::~Surface() { delete pimpl; }

CubemapSurface::~CubemapSurface() { delete pimpl; }

Surface::Surface(DeviceSurface* pimpl, size_t width, size_t height) : pimpl(pimpl), _width(width), _height(height) {}

CubemapSurface::CubemapSurface(DeviceCubemapSurface* pimpl, size_t length) : pimpl(pimpl), _length(length) {}

DeviceSurface::~DeviceSurface() {
  cudaDestroyTextureObject(tex_);
  cudaDestroySurfaceObject(surface_);
  if (ownsArray_) {
    cudaFreeArray(array_);
    deviceStats.deletePtr(array_);
  }
}

DeviceSurface& Surface::get() {
  assert(pimpl);
  return *pimpl;
}
const DeviceSurface& Surface::get() const {
  assert(pimpl);
  return *pimpl;
}

DeviceCubemapSurface& CubemapSurface::get() {
  assert(pimpl);
  return *pimpl;
}
const DeviceCubemapSurface& CubemapSurface::get() const {
  assert(pimpl);
  return *pimpl;
}

bool Surface::operator==(const Surface& other) const {
  if (pimpl && other.pimpl) {
    return *pimpl == *other.pimpl;
  }
  return !pimpl && !other.pimpl;
}

}  // namespace GPU

namespace Core {

Status SourceOpenGLSurface::Pimpl::acquire() {
  acquired = true;
  return CUDA_ERROR(cudaGraphicsMapResources(1, &image, stream.get()));
}

Status SourceOpenGLSurface::Pimpl::release() {
  if (acquired) {
    acquired = false;
    return CUDA_ERROR(cudaGraphicsUnmapResources(1, &image, stream.get()));
  }
  return Status::OK();
}

Status PanoOpenGLSurface::Pimpl::acquire() {
  acquired = true;
  return CUDA_ERROR(cudaGraphicsMapResources(1, &image, stream.get()));
}

Status PanoOpenGLSurface::Pimpl::release() {
  if (acquired) {
    acquired = false;
    return CUDA_ERROR(cudaGraphicsUnmapResources(1, &image, stream.get()));
  }
  return Status::OK();
}

Status CubemapOpenGLSurface::Pimpl::acquire() {
  for (int i = 0; i < 6; ++i) {
    FAIL_RETURN(CUDA_ERROR(cudaGraphicsMapResources(1, &resources[i], stream.get())));
  }
  return Status::OK();
}

Status CubemapOpenGLSurface::Pimpl::release() {
  for (int i = 0; i < 6; ++i) {
    FAIL_RETURN(CUDA_ERROR(cudaGraphicsUnmapResources(1, &resources[i], stream.get())));
  }
  return Status::OK();
}

Potential<PanoOpenGLSurface::Pimpl> PanoOpenGLSurface::Pimpl::create(cudaGraphicsResource_t image,
                                                                     GPU::Buffer<uint32_t> buffer,
                                                                     GPU::Surface* remapSurf, size_t w, size_t h) {
  PotentialValue<GPU::Stream> stream = GPU::Stream::create();
  if (stream.ok()) {
    return Potential<Pimpl>(new Pimpl(image, buffer, remapSurf, w, h, stream.value()));
  } else {
    return Potential<Pimpl>(stream.status());
  }
}

Potential<CubemapOpenGLSurface::Pimpl> CubemapOpenGLSurface::Pimpl::create(
    cudaGraphicsResource_t* resources, GPU::Buffer<uint32_t>* buffers, GPU::Buffer<uint32_t> buffer,
    GPU::CubemapSurface* cubemap, GPU::Buffer<uint32_t> tmp, size_t w, bool equiangular) {
  PotentialValue<GPU::Stream> stream = GPU::Stream::create();
  if (stream.ok()) {
    return Potential<Pimpl>(new Pimpl(equiangular, resources, buffers, buffer, cubemap, tmp, w, stream.value()));
  } else {
    return Potential<Pimpl>(stream.status());
  }
}

PanoOpenGLSurface::Pimpl::~Pimpl() { cudaGraphicsUnregisterResource(image); }

CubemapOpenGLSurface::Pimpl::~Pimpl() {
  for (int i = 0; i < 6; ++i) {
    cudaGraphicsUnregisterResource(resources[i]);
  }
}

Potential<SourceOpenGLSurface::Pimpl> SourceOpenGLSurface::Pimpl::create(cudaGraphicsResource_t image,
                                                                         std::unique_ptr<GPU::Surface> surf) {
  PotentialValue<GPU::Stream> stream = GPU::Stream::create();
  if (stream.ok()) {
    return Potential<Pimpl>(new Pimpl(image, surf.release(), stream.value()));
  } else {
    return Potential<Pimpl>(stream.status());
  }
}

SourceOpenGLSurface::Pimpl::~Pimpl() { cudaGraphicsUnregisterResource(image); }

}  // namespace Core
}  // namespace VideoStitch
