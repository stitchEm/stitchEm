// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#ifdef VS_OPENCL
#error This file is part of the CUDA backend. It is not supposed to be included in libvideostitch-OpenCL.
#endif

#include "gpu/allocator.hpp"
#include "gpu/surface.hpp"

#include <cuda_runtime.h>

namespace VideoStitch {
namespace GPU {

template <typename T>
class Buffer;
class Stream;

class DeviceSurface {
 public:
  DeviceSurface(cudaArray_t arr, cudaTextureObject_t tex, cudaSurfaceObject_t surf, bool ownsArray)
      : array_(arr), tex_(tex), surface_(surf), ownsArray_(ownsArray) {}

  ~DeviceSurface();

  cudaSurfaceObject_t surface() const { return surface_; }

  cudaTextureObject_t texture() const { return tex_; }

  cudaArray_t array() const { return array_; }

  bool operator==(const DeviceSurface& other) const { return array_ == other.array_; }

 private:
  cudaArray_t array_;
  cudaTextureObject_t tex_;
  cudaSurfaceObject_t surface_;
  bool ownsArray_;

  template <typename T>
  friend Status memcpyAsync(Surface&, Buffer<const T>, const Stream&);
  template <typename T>
  friend Status memcpyAsync(Buffer<T>, Surface&, const Stream&);
  template <typename T>
  friend Status memcpyAsync(T*, Surface&, const Stream&);
  friend Status memcpyAsync(Surface& dst, uint32_t* src, const Stream& stream);

  friend Status memcpyBlocking(Surface&, Buffer<const uint32_t>);
  friend Status memcpyBlocking(Buffer<uint32_t>, Surface&);
  template <typename T>
  friend Status memcpyBlocking(T*, Surface&);
  friend Status memcpyBlocking(Surface& dst, uint32_t* src);
};

class DeviceCubemapSurface : public DeviceSurface {
 public:
  DeviceCubemapSurface(cudaArray_t arr, cudaTextureObject_t tex, cudaSurfaceObject_t surf, bool ownsArray)
      : DeviceSurface(arr, tex, surf, ownsArray) {}
};
}  // namespace GPU

namespace Core {
class SourceOpenGLSurface::Pimpl : public SourceSurface::Pimpl {
 public:
  static Potential<Pimpl> create(cudaGraphicsResource_t, std::unique_ptr<GPU::Surface>);
  virtual ~Pimpl();

  virtual Status acquire();
  virtual Status release();

 private:
  Pimpl(cudaGraphicsResource_t image, GPU::Surface* surf, GPU::Stream str)
      : SourceSurface::Pimpl(surf, str), image(image) {}

  cudaGraphicsResource_t image;
};

class PanoOpenGLSurface::Pimpl : public PanoPimpl {
 public:
  static Potential<Pimpl> create(cudaGraphicsResource_t, GPU::Buffer<uint32_t>, GPU::Surface*, size_t w, size_t h);
  virtual ~Pimpl();

  virtual Status acquire();
  virtual Status release();

 private:
  Pimpl(cudaGraphicsResource_t image, GPU::Buffer<uint32_t> buffer, GPU::Surface* remapSurface, size_t w, size_t h,
        GPU::Stream str)
      : PanoPimpl(str, buffer, remapSurface, w, h), image(image) {}

  cudaGraphicsResource_t image;
};

class CubemapOpenGLSurface::Pimpl : public CubemapPimpl {
 public:
  static Potential<Pimpl> create(cudaGraphicsResource_t*, GPU::Buffer<uint32_t>*, GPU::Buffer<uint32_t>,
                                 GPU::CubemapSurface*, GPU::Buffer<uint32_t>, size_t, bool equiangular);
  virtual ~Pimpl();

  virtual Status acquire();
  virtual Status release();

 private:
  Pimpl(bool equiangular, cudaGraphicsResource_t* r, GPU::Buffer<uint32_t>* buffers, GPU::Buffer<uint32_t> buffer,
        GPU::CubemapSurface* cubemap, GPU::Buffer<uint32_t> tmp, size_t w, GPU::Stream str)
      : CubemapPimpl(equiangular, str, buffers, buffer, cubemap, tmp, w) {
    for (int i = 0; i < 6; ++i) {
      resources[i] = r[i];
    }
  }

  cudaGraphicsResource_t resources[6];
};

}  // namespace Core
}  // namespace VideoStitch
