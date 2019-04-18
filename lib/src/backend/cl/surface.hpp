// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/allocator.hpp"

#include "opencl.h"

namespace VideoStitch {
namespace GPU {

class DeviceSurface {
 public:
  explicit DeviceSurface(cl_mem im, bool ownsImage) : image(im), ownsImage(ownsImage) {}

  ~DeviceSurface();

  cl_mem& raw() { return image; }

  operator cl_mem() const { return image; }

  operator cl_mem&() { return image; }

  bool operator==(const DeviceSurface& other) const { return image == other.image; }

 private:
  cl_mem image;
  bool ownsImage;
};

class DeviceCubemapSurface {
 public:
  ~DeviceCubemapSurface();
};

}  // namespace GPU

namespace Core {

class SourceOpenGLSurface::Pimpl : public SourceSurface::Pimpl {
 public:
  static Potential<Pimpl> create(GPU::Surface*);
  virtual ~Pimpl() {}

  virtual Status acquire();
  virtual Status release();

 private:
  Pimpl(GPU::Surface* surf, GPU::Stream str) : SourceSurface::Pimpl(surf, str) {}
};

class PanoOpenGLSurface::Pimpl : public PanoPimpl {
 public:
  static Potential<Pimpl> create(GPU::Buffer<uint32_t>, GPU::Surface*, size_t w, size_t h);
  virtual ~Pimpl() {}

  virtual Status acquire();
  virtual Status release();

 private:
  Pimpl(GPU::Stream str, GPU::Buffer<uint32_t> buffer, GPU::Surface* remapSurf, size_t w, size_t h)
      : PanoPimpl(str, buffer, remapSurf, w, h) {}
};

class CubemapOpenGLSurface::Pimpl : public CubemapPimpl {
 public:
  static Potential<Pimpl> create(GPU::Buffer<uint32_t>*, GPU::Buffer<uint32_t>, GPU::CubemapSurface*,
                                 GPU::Buffer<uint32_t>, size_t, bool equiangular);
  virtual ~Pimpl() {}

  virtual Status acquire();
  virtual Status release();

 private:
  Pimpl(bool equiangular, GPU::Buffer<uint32_t>* buffers, GPU::Buffer<uint32_t> buffer, GPU::CubemapSurface* remapSurf,
        GPU::Buffer<uint32_t> tmp, size_t w, GPU::Stream str)
      : CubemapPimpl(equiangular, str, buffers, buffer, remapSurf, tmp, w) {}
};

}  // namespace Core
}  // namespace VideoStitch
