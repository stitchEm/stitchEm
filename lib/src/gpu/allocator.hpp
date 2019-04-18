// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/allocator.hpp"

#include "buffer.hpp"
#include "surface.hpp"
#include "stream.hpp"
#include "uniqueBuffer.hpp"

#include <condition_variable>

namespace VideoStitch {

template <typename T>
class Matrix33;

namespace Core {
class ImageMapping;
class PanoRemapper;
class PanoDefinition;
class ImageMerger;

class SourceSurface::Pimpl {
 public:
  GPU::Surface* surface;
  GPU::Stream stream;

  static Potential<Pimpl> create(GPU::Surface*);
  virtual ~Pimpl();

  void acquireWriter();
  void acquireReader();
  void releaseWriter();
  void releaseReader();

  size_t getWidth() const;
  size_t getHeight() const;

  virtual Status acquire() { return Status::OK(); }
  virtual Status release() { return Status::OK(); }

 protected:
  Pimpl(GPU::Surface*, GPU::Stream);

  std::mutex mutex;
  std::condition_variable cv;
  int renderers = 0;
  bool stitcher = false;
  bool acquired = false;

  friend class OffscreenAllocator;
  friend class OpenGLAllocator;
};

class PanoSurface::Pimpl {
 public:
  GPU::Buffer<uint32_t> buffer;
  bool externalAlloc;
  GPU::Stream stream;

  virtual ~Pimpl();

  void acquireWriter();
  void acquireReader();
  void releaseWriter();
  void releaseReader();

  size_t getWidth() const { return width; }
  size_t getHeight() const { return height; }

  virtual Status acquire() { return Status::OK(); }

  virtual Status release() { return Status::OK(); }

  virtual Status reset(const Core::ImageMerger*) { return Status::OK(); }
  virtual Status reproject(const Core::PanoDefinition&, const Matrix33<double>& perspective,
                           const Core::ImageMerger*) = 0;
  virtual Status warp(Core::ImageMapping*, frameid_t, const Core::PanoDefinition&, GPU::Stream&) = 0;
  virtual Status blend(const Core::PanoDefinition&, const Core::ImageMapping&, bool firstMerger, GPU::Stream&) = 0;
  virtual Status flatten() = 0;
  virtual Status reconstruct(const Core::PanoDefinition&, const Core::ImageMapping&, GPU::Stream&,
                             bool final = true) = 0;

 protected:
  Pimpl(GPU::Stream, GPU::Buffer<uint32_t>, size_t w, size_t h);

  size_t width, height;

  std::mutex mutex;
  std::condition_variable cv;
  int renderers = 0;
  bool stitcher = false;
  bool acquired = false;

  friend class OffscreenAllocator;
  friend class OpenGLAllocator;
};

class PanoPimpl : public PanoSurface::Pimpl {
 public:
  GPU::UniqueBuffer<uint32_t> progressivePbo;

  static Potential<PanoPimpl> create(GPU::Buffer<uint32_t>, GPU::Surface*, size_t w, size_t h);
  virtual ~PanoPimpl();

  Status reset(const Core::ImageMerger*) override;
  Status reproject(const Core::PanoDefinition&, const Matrix33<double>& perspective, const Core::ImageMerger*) override;
  Status warp(Core::ImageMapping*, frameid_t, const Core::PanoDefinition&, GPU::Stream&) override;
  Status blend(const Core::PanoDefinition&, const Core::ImageMapping&, bool firstMerger, GPU::Stream&) override;
  Status flatten() override;
  Status reconstruct(const Core::PanoDefinition&, const Core::ImageMapping&, GPU::Stream&, bool final) override;

 protected:
  PanoPimpl(GPU::Stream, GPU::Buffer<uint32_t>, GPU::Surface*, size_t w, size_t h);

  GPU::Surface* remapBuffer = nullptr;

  friend class OffscreenAllocator;
  friend class OpenGLAllocator;
};

class CubemapSurface::Pimpl : public PanoSurface::Pimpl {
 public:
  Pimpl(GPU::Stream stream, GPU::Buffer<uint32_t> buffer, size_t w)
      : PanoSurface::Pimpl(stream, buffer, 3 * w, 2 * w), length(w) {}

  size_t getLength() const { return length; }

 protected:
  size_t length;
};

class CubemapPimpl : public CubemapSurface::Pimpl {
 public:
  GPU::Buffer<uint32_t> buffers[6];

  Layout layout = ROT;
  bool equiangular;  // runtime type tag, cause I don't want to duplicate or template the whole type hierarchy

  // static Potential<Pimpl> create(size_t w);
  virtual ~CubemapPimpl();

  Status reproject(const Core::PanoDefinition&, const Matrix33<double>& perspective, const Core::ImageMerger*) override;
  Status warp(Core::ImageMapping*, frameid_t, const Core::PanoDefinition&, GPU::Stream&) override;
  Status blend(const Core::PanoDefinition&, const Core::ImageMapping&, bool firstMerger, GPU::Stream&) override;
  Status flatten() override;
  Status reconstruct(const Core::PanoDefinition&, const Core::ImageMapping&, GPU::Stream&, bool final) override;

 protected:
  CubemapPimpl(bool equiangular, GPU::Stream, GPU::Buffer<uint32_t>*, GPU::Buffer<uint32_t>, GPU::CubemapSurface*,
               GPU::Buffer<uint32_t>, size_t w);

  GPU::CubemapSurface* remapBuffer;
  GPU::Buffer<uint32_t> tmp;

  friend class OffscreenAllocator;
  friend class OpenGLAllocator;
};
}  // namespace Core
}  // namespace VideoStitch
