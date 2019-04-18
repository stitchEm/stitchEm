// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "../../image/colorArray.hpp"

#include "gpu/2dBuffer.hpp"
#include "gpu/uniqueBuffer.hpp"
#include "gpu/stream.hpp"
#include "gpu/allocator.hpp"

#include "libvideostitch/output.hpp"
#include "libvideostitch/profile.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/stitchOutput.hpp"

#include <condition_variable>
#include <map>
#include <mutex>
#include <vector>

namespace VideoStitch {
namespace Core {
/**
 * Represents the data for a single format.
 */
struct FormatBuffer;

/**
 * A frame buffer.
 *
 * Can be host- or device- backed.
 * Currently instantiated for 32 bits RGBA, with 2 variants:
 * - 2 alpha, 10 color bits for panorama internal format.
 * - 8 bits for each components for source frames.
 * Can represent either panorama or input frames.
 *
 * Possible output formats are:
 *  - RGBA
 *  - RGB
 *  - planarYV12
 *
 * fill() marks the end of commands queueing.
 * The frame can then be synchronized and presented to the user's callback.
 * Thread-safe.
 */
template <typename Writer, typename Surface>
class FrameBuffer {
 public:
  typedef PanoOpenGLSurface GLSurface;

  static Potential<FrameBuffer<Writer, Surface>> create(std::shared_ptr<Surface> surf,
                                                        const std::vector<std::shared_ptr<Writer>>& writers);
  static Potential<FrameBuffer<Writer, Surface>> create(std::shared_ptr<Surface> surf, std::shared_ptr<Writer> writer);
  static Potential<FrameBuffer<Writer, Surface>> create(std::shared_ptr<Surface> surf);

  FrameBuffer& operator=(FrameBuffer&&);
  virtual ~FrameBuffer();

  /**
   * Fills the buffer asynchronously.
   */
  Status pushVideo();

  Status registerWriters(const std::vector<std::shared_ptr<Writer>>& writers) {
    for (auto w : writers) {
      FAIL_RETURN(registerWriter(w));
    }
    return Status::OK();
  }
  Status registerWriter(std::shared_ptr<Writer>);

  void streamSynchronize() {
    surf->pimpl->releaseWriter();
    if (!surf->pimpl->stream.synchronize().ok()) {
      Logger::get(Logger::Error) << "GPU error when stitching frame at " << date << " useconds " << std::endl;
    } else {
      const uint64_t duration = timer.elapsed();
      Logger::get(Logger::Verbose) << "stitched frame at " << date << " useconds "
                                   << " : " << duration / 1000 << " ms" << std::endl;
      date = -1;
    }
  }

  std::shared_ptr<Surface> getSurface() const { return surf; }

  std::shared_ptr<GLSurface> getOpenGLSurface() {
    if (oglSurf == nullptr) {
      Potential<PanoOpenGLSurface> osurf = OpenGLAllocator::createPanoSurface(
          surf->getWidth(), surf->getHeight(), OpenGLAllocator::BufferAllocType::ReadOnly);
      if (osurf.ok()) {
        oglSurf = std::shared_ptr<GLSurface>(osurf.release());
      } else {
        Logger::get(Logger::Error) << "Can not allocate OpenGL shared buffer on GPU." << std::endl;
      }
    }
    return oglSurf;
  }

  void streamOpenGLSynchronize() {
    oglSurf->pimpl->releaseWriter();
    if (!oglSurf->pimpl->stream.synchronize().ok()) {
      Logger::get(Logger::Error) << "GPU error when stitching frame at " << date << " useconds " << std::endl;
    } else {
      const uint64_t duration = timer.elapsed();
      Logger::get(Logger::Verbose) << "stitched frame at " << date << " useconds "
                                   << " : " << duration / 1000 << " ms" << std::endl;
      date = -1;
    }
  }

  Frame getFrame(PixelFormat, AddressSpace, size_t);

  void releaseFrame() {
    surf->pimpl->releaseWriter();
    surf->pimpl->release();
  }

 protected:
  explicit FrameBuffer(std::shared_ptr<Surface>);
  FrameBuffer(FrameBuffer<Writer, Surface>&&);

  mtime_t date;
  Util::SimpleTimer timer;

  mutable std::mutex mutex;  // Protect formatBuffers
  typedef std::map<VideoStitch::PixelFormat, FormatBuffer*> FormatBufferMap;
  FormatBufferMap formatBuffers;

  std::shared_ptr<Surface> surf;
  std::shared_ptr<GLSurface> oglSurf;
};

class SourceFrameBuffer : public FrameBuffer<Output::VideoWriter, SourceSurface> {
 public:
  typedef SourceSurface Surface;
  typedef SourceOpenGLSurface GLSurface;
  typedef SourceRenderer Renderer;

  static Potential<SourceFrameBuffer> create(std::shared_ptr<SourceSurface> surf,
                                             const std::vector<std::shared_ptr<Output::VideoWriter>>& writers);
  static Potential<SourceFrameBuffer> create(std::shared_ptr<SourceSurface> surf,
                                             std::shared_ptr<Output::VideoWriter> writer);
  static Potential<SourceFrameBuffer> create(std::shared_ptr<SourceSurface> surf);

  SourceFrameBuffer& operator=(SourceFrameBuffer&&);
  virtual ~SourceFrameBuffer() {}

  GPU::Surface& acquireFrame(mtime_t d, GPU::Stream& str) {
    if (date == -1) {
      date = d;
      timer.reset();
    }
    surf->pimpl->acquireWriter();
    surf->pimpl->acquire();
    str = surf->pimpl->stream;
    return *surf->pimpl->surface;
  }

  // Fills the buffer asynchronously.
  Status pushVideo();
  Status pushOpenGLVideo();

 private:
  explicit SourceFrameBuffer(std::shared_ptr<SourceSurface> s) : FrameBuffer(s) {}
  SourceFrameBuffer(SourceFrameBuffer&&);
};

template <typename Writer>
class StitchFrameBuffer : public FrameBuffer<Writer, PanoSurface> {
 public:
  typedef PanoSurface Surface;
  typedef PanoOpenGLSurface GLSurface;
  typedef PanoRenderer Renderer;

  static Potential<StitchFrameBuffer> create(std::shared_ptr<PanoSurface> surf,
                                             const std::vector<std::shared_ptr<Writer>>& writers);
  static Potential<StitchFrameBuffer> create(std::shared_ptr<PanoSurface> surf, std::shared_ptr<Writer> writer);
  static Potential<StitchFrameBuffer> create(std::shared_ptr<PanoSurface> surf);

  StitchFrameBuffer& operator=(StitchFrameBuffer&&);
  virtual ~StitchFrameBuffer() {}

  PanoSurface& acquireFrame(mtime_t d) {
    if (this->date == -1) {
      this->date = d;
      this->timer.reset();
    }
    this->surf->pimpl->acquireWriter();
    this->surf->pimpl->acquire();
    return *this->surf;
  }

  // Fills the frame buffer asynchronously.
  Status pushVideo();
  Status pushOpenGLVideo();

 private:
  explicit StitchFrameBuffer(std::shared_ptr<PanoSurface> s) : FrameBuffer<Writer, PanoSurface>(s) {}
  StitchFrameBuffer(StitchFrameBuffer&&);
};

typedef StitchFrameBuffer<Output::VideoWriter> PanoFrameBuffer;
typedef StitchFrameBuffer<Output::StereoWriter> StereoFrameBuffer;

}  // namespace Core
}  // namespace VideoStitch
