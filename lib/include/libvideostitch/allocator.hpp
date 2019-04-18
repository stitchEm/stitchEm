// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "status.hpp"

#include <memory>

namespace VideoStitch {

namespace GPU {
class Overlayer;
}

namespace Core {

class SourceRenderer;
class PanoRenderer;
class PanoOpenGLSurface;

class VS_EXPORT SourceSurface : public std::enable_shared_from_this<SourceSurface> {
 public:
  class Pimpl;

  virtual ~SourceSurface();

  size_t getWidth() const;
  size_t getHeight() const;
  size_t sourceId;

  void acquire();
  void release();

  Pimpl* const pimpl;
  explicit SourceSurface(Pimpl*);

  virtual void accept(std::shared_ptr<SourceRenderer>, mtime_t);
  void accept(const std::shared_ptr<GPU::Overlayer>&, std::shared_ptr<PanoOpenGLSurface>, mtime_t) {}

 private:
  SourceSurface(const SourceSurface&);
  const SourceSurface& operator=(const SourceSurface&);
};

class VS_EXPORT PanoSurface : public std::enable_shared_from_this<PanoSurface> {
 public:
  class Pimpl;

  virtual ~PanoSurface();

  size_t getWidth() const;
  size_t getHeight() const;

  void acquire();
  void release();

  Pimpl* const pimpl;
  explicit PanoSurface(Pimpl*);

  virtual void accept(std::shared_ptr<PanoRenderer>, mtime_t);
  virtual void accept(const std::shared_ptr<GPU::Overlayer>&, std::shared_ptr<PanoOpenGLSurface>, mtime_t);

 private:
  PanoSurface(const PanoSurface&);
  const PanoSurface& operator=(const PanoSurface&);
};

class VS_EXPORT SourceOpenGLSurface : public SourceSurface {
 public:
  class Pimpl;

  int texture;

  explicit SourceOpenGLSurface(Pimpl*);
  virtual ~SourceOpenGLSurface();

  virtual void accept(std::shared_ptr<SourceRenderer>, mtime_t);

 private:
  SourceOpenGLSurface(const SourceOpenGLSurface&);
  const SourceOpenGLSurface& operator=(const SourceOpenGLSurface&);
};

class VS_EXPORT PanoOpenGLSurface : public PanoSurface {
 public:
  class Pimpl;

  int pixelbuffer;

  explicit PanoOpenGLSurface(Pimpl*);
  virtual ~PanoOpenGLSurface();

  virtual void accept(std::shared_ptr<PanoRenderer>, mtime_t);
  virtual void accept(const std::shared_ptr<GPU::Overlayer>&, std::shared_ptr<PanoOpenGLSurface>, mtime_t);

 private:
  PanoOpenGLSurface(const PanoOpenGLSurface&);
  const PanoOpenGLSurface& operator=(const PanoOpenGLSurface&);
};

enum Layout { YOUTUBE = 0, ROT = 1 };

class VS_EXPORT CubemapSurface : public PanoSurface {
 public:
  class Pimpl;

  explicit CubemapSurface(Pimpl*);
  virtual ~CubemapSurface();

  size_t getLength() const;

  virtual void accept(std::shared_ptr<PanoRenderer>, mtime_t);
  virtual void accept(const std::shared_ptr<GPU::Overlayer>&, std::shared_ptr<PanoOpenGLSurface>, mtime_t);

 private:
  CubemapSurface(const CubemapSurface&);
  const CubemapSurface& operator=(const CubemapSurface&);
};

class VS_EXPORT CubemapOpenGLSurface : public CubemapSurface {
 public:
  class Pimpl;

  int faces[6];

  CubemapOpenGLSurface(Pimpl*, int* faces);
  virtual ~CubemapOpenGLSurface();

  virtual void accept(std::shared_ptr<PanoRenderer>, mtime_t);
  virtual void accept(const std::shared_ptr<GPU::Overlayer>&, std::shared_ptr<PanoOpenGLSurface>, mtime_t);

 private:
  CubemapOpenGLSurface(const CubemapOpenGLSurface&);
  const CubemapOpenGLSurface& operator=(const CubemapOpenGLSurface&);
};

class VS_EXPORT OffscreenAllocator {
 public:
  static Potential<SourceSurface> createAlphaSurface(size_t width, size_t height, const char* name);
  static Potential<SourceSurface> createSourceSurface(size_t width, size_t height, const char* name);
  static Potential<SourceSurface> createDepthSurface(size_t width, size_t height, const char* name);
  static Potential<SourceSurface> createCoordSurface(size_t width, size_t height, const char* name);
  static Potential<PanoSurface> createPanoSurface(size_t width, size_t height, const char* name);
  static Potential<CubemapSurface> createCubemapSurface(size_t width, const char* name, bool equiangular);
};

class VS_EXPORT OpenGLAllocator {
 public:
  enum class BufferAllocType { ReadWrite, ReadOnly, WriteOnly };

  static Potential<SourceOpenGLSurface> createSourceSurface(size_t width, size_t height);
  static Potential<PanoOpenGLSurface> createPanoSurface(size_t width, size_t height,
                                                        BufferAllocType flag = BufferAllocType::WriteOnly);
  static Potential<CubemapOpenGLSurface> createCubemapSurface(size_t width, bool equiangular,
                                                              BufferAllocType flag = BufferAllocType::WriteOnly);
};

class VS_EXPORT SourceRenderer {
 public:
  virtual std::string getName() const = 0;
  virtual void render(std::shared_ptr<SourceOpenGLSurface>, mtime_t) = 0;
};

class VS_EXPORT PanoRenderer {
 public:
  virtual std::string getName() const = 0;
  virtual void render(std::shared_ptr<PanoOpenGLSurface>, mtime_t) = 0;
  virtual void renderCubemap(std::shared_ptr<CubemapOpenGLSurface>, mtime_t) = 0;
  virtual void renderEquiangularCubemap(std::shared_ptr<CubemapOpenGLSurface>, mtime_t) = 0;
};

}  // namespace Core
}  // namespace VideoStitch
