// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "frameBuffer.hpp"

#include "common/container.hpp"
#include "gpu/image/downsampler.hpp"
#include "image/unpack.hpp"

#include "gpu/buffer.hpp"
#include "gpu/hostBuffer.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/util.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/profile.hpp"

#include <algorithm>
#include <iostream>
#include <thread>

// TODO_OPENCL_IMPL
// backend dependent code needs to be moved to the backend itself
#ifdef VS_OPENCL
#include <backend/cl/deviceBuffer.hpp>
#else
#include <backend/cuda/deviceBuffer.hpp>
#include <backend/cuda/deviceStream.hpp>
#endif

namespace VideoStitch {
namespace Core {

/**
 * The product of the stitching process is a panorama in full-scale, RGBA210 pixel format on the GPU.
 *
 * Yet the user callbacks can consume:
 * - non-RGBA210 panorama (eg. RGBA8888 !)
 * - downsampled panorama (eg. panorama stitched in 4K but streamed in 2K on the network)
 * - host-based buffer (eg. callbacks which are not OpenGL renderers and hardware encoders)
 *
 * The conversion process is as follow:
 * |panorama| -- colorspace conversion --> |1| -- downsampling --> |2| -- upload to host --> |final panorama|
 *
 * The design enforces that by covering all operations needed through:
 * - a map colorspace -> FormatBuffer inside the FrameBuffer
 * - a map downsampling ratio -> DownsamplingBuffer inside the FormatBuffer
 *
 * The conversion is done through FrameBuffer::pushVideo(), which will call FormatBuffer::pushVideo() for every
 * colorspace requested by the user. Then FormatBuffer::pushVideo() will call downsample() then
 * DownsamplingBuffer::pushVideo() for every downsampling ratio requested by the user. Inside it MemcpyAsync() will be
 * called in case a host-based panorama has been requested too.
 */

struct DownsamplingBuffer {
  DownsamplingBuffer(int32_t w, int32_t h, int ratio, PixelFormat fmt) : ratio(ratio), fmt(fmt), width(w), height(h) {}

  ~DownsamplingBuffer() {
    hostBuffers[0].release();
    hostBuffers[1].release();
    hostBuffers[2].release();
    downBuf[0].release();
    downBuf[1].release();
    downBuf[2].release();
  }

  Status pushVideo(GPU::Buffer2D* data, GPU::Stream stream) {
    if (ratio > 1) {
      Image::downsample(fmt, data, downBuf, stream);
    } else {
      downBuf[0].swap(data[0]);
      downBuf[1].swap(data[1]);
      downBuf[2].swap(data[2]);
    }

    // upload to host
    if (hostBuffers[0].hostPtr()) {
      PROPAGATE_FAILURE_STATUS(GPU::memcpyAsync(hostBuffers[0].hostPtr(), downBuf[0], stream));
    }
    if (hostBuffers[1].hostPtr()) {
      PROPAGATE_FAILURE_STATUS(GPU::memcpyAsync(hostBuffers[1].hostPtr(), downBuf[1], stream));
    }
    if (hostBuffers[2].hostPtr()) {
      PROPAGATE_FAILURE_STATUS(GPU::memcpyAsync(hostBuffers[2].hostPtr(), downBuf[2], stream));
    }
    ready = true;
    return Status::OK();
  }

#define ALLOCATE_DOWNBUFFER(host, dev, width, height, writer)                                             \
  {                                                                                                       \
    if (writer->getExpectedOutputBufferType() == Host) {                                                  \
      auto potHostBuffer =                                                                                \
          GPU::HostBuffer<unsigned char>::allocate(width * height, "Panorama Frame", GPUHostAllocPinned); \
      PROPAGATE_FAILURE_STATUS(potHostBuffer.status());                                                   \
      host.release();                                                                                     \
      host = potHostBuffer.value();                                                                       \
    }                                                                                                     \
    auto potDevBuffer = GPU::Buffer2D::allocate(width, height, "Panorama Frame");                         \
    PROPAGATE_FAILURE_STATUS(potDevBuffer.status());                                                      \
    dev.release();                                                                                        \
    dev = potDevBuffer.value();                                                                           \
  }

  template <typename Writer>
  Status registerWriter(std::shared_ptr<Writer> writer) {
    switch (fmt) {
      case RGBA:
      case BGRU:
      case F32_C1:
        ALLOCATE_DOWNBUFFER(hostBuffers[0], downBuf[0], width * 4, height, writer)
        break;
      case RGB:
      case BGR:
        ALLOCATE_DOWNBUFFER(hostBuffers[0], downBuf[0], width * 3, height, writer)
        break;
      case UYVY:
      case YUY2:
      case Grayscale16:
        ALLOCATE_DOWNBUFFER(hostBuffers[0], downBuf[0], width * 2, height, writer)
        break;
      case YV12:
      case DEPTH:
        ALLOCATE_DOWNBUFFER(hostBuffers[0], downBuf[0], width, height, writer)
        ALLOCATE_DOWNBUFFER(hostBuffers[1], downBuf[1], width / 2, height / 2, writer)
        ALLOCATE_DOWNBUFFER(hostBuffers[2], downBuf[2], width / 2, height / 2, writer)
        break;
      case NV12:
        ALLOCATE_DOWNBUFFER(hostBuffers[0], downBuf[0], width, height, writer)
        ALLOCATE_DOWNBUFFER(hostBuffers[1], downBuf[1], width, height / 2, writer)
        break;
      case YUV422P10:
        ALLOCATE_DOWNBUFFER(hostBuffers[0], downBuf[0], width * 2, height, writer)
        ALLOCATE_DOWNBUFFER(hostBuffers[1], downBuf[1], width, height, writer)
        ALLOCATE_DOWNBUFFER(hostBuffers[2], downBuf[2], width, height, writer)
        break;
      default:
        assert(false);
        return {Origin::Stitcher, ErrType::ImplementationError, "Unsupported colorspace for downsampling"};
    }
    return Status::OK();
  }

  Frame getFrame(AddressSpace addr) {
    Frame frame = {{nullptr, nullptr, nullptr}, {0, 0, 0}, width, height, -1, Unknown};
    switch (addr) {
      case Host:
        frame.planes[0] = hostBuffers[0].hostPtr();
        frame.pitches[0] = hostBuffers[0].byteSize() / frame.height;
        frame.planes[1] = hostBuffers[1].hostPtr();
        frame.pitches[1] = hostBuffers[1].byteSize() / frame.height;
        frame.planes[2] = hostBuffers[2].hostPtr();
        frame.pitches[2] = hostBuffers[2].byteSize() / frame.height;
        break;
      case Device:
        frame.planes[0] = downBuf[0].devicePtr();
        frame.pitches[0] = downBuf[0].getPitch();
        frame.planes[1] = downBuf[1].devicePtr();
        frame.pitches[1] = downBuf[1].getPitch();
        frame.planes[2] = downBuf[2].devicePtr();
        frame.pitches[2] = downBuf[2].getPitch();
        break;
    }
    return frame;
  }

  // This marker is used to avoid a race condition when a frame has been completely scheduled
  // for processing, but a new writer is registered while the GPU is being synchronized.
  // In that case, the frame buffer might miss eg. a new colorspace while the StitchOutput
  // expects to find it for the newly registered callback.
  // Thus, getBuffer might return null in this specific circumstance
  bool ready = false;
  int ratio;
  PixelFormat fmt;
  GPU::Buffer2D downBuf[3];
  GPU::HostBuffer<unsigned char> hostBuffers[3];
  int32_t width, height;

 private:
  DownsamplingBuffer(const DownsamplingBuffer&);
  DownsamplingBuffer& operator=(const DownsamplingBuffer&);
};

// ---------------------- Colorspace conversion

struct FormatBuffer {
  FormatBuffer(PixelFormat fmt, int64_t width, int64_t height) : pxFmt(fmt), width(width), height(height) {}

  ~FormatBuffer() {
    colorconvBuf[0].release();
    colorconvBuf[1].release();
    colorconvBuf[2].release();
    deleteAllValues(downsamplers);
  }

  /**
   * Reads back the given device buffer and put in in the format buffer.
   */
  template <typename Surface>
  Status pushVideo(Surface& data, GPU::Stream stream) {
    // convert colorspace
    switch (pxFmt) {
      case VideoStitch::PixelFormat::RGBA:
        PROPAGATE_FAILURE_STATUS(Image::unpackRGBA(colorconvBuf[0], data, width, height, stream));
        break;
      case VideoStitch::PixelFormat::F32_C1:
        PROPAGATE_FAILURE_STATUS(Image::unpackF32C1(colorconvBuf[0], data, width, height, stream));
        break;
      case VideoStitch::PixelFormat::Grayscale16:
        PROPAGATE_FAILURE_STATUS(Image::unpackGrayscale16(colorconvBuf[0], data, width, height, stream));
        break;
      case VideoStitch::PixelFormat::DEPTH:
        PROPAGATE_FAILURE_STATUS(
            Image::unpackDepth(colorconvBuf[0], colorconvBuf[1], colorconvBuf[2], data, width, height, stream));
        break;
      case VideoStitch::PixelFormat::RGB: {
        PROPAGATE_FAILURE_STATUS(Image::unpackRGB(colorconvBuf[0], data, width, height, stream));
        break;
      }
      case VideoStitch::PixelFormat::YV12: {
        PROPAGATE_FAILURE_STATUS(
            Image::unpackYV12(colorconvBuf[0], colorconvBuf[1], colorconvBuf[2], data, width, height, stream));
        break;
      }
      case VideoStitch::PixelFormat::NV12: {
        PROPAGATE_FAILURE_STATUS(Image::unpackNV12(colorconvBuf[0], colorconvBuf[1], data, width, height, stream));
        break;
      }
      case VideoStitch::PixelFormat::UYVY: {
        PROPAGATE_FAILURE_STATUS(Image::unpackUYVY(colorconvBuf[0], data, width, height, stream));
        break;
      }
      case VideoStitch::PixelFormat::YUY2: {
        PROPAGATE_FAILURE_STATUS(Image::unpackYUY2(colorconvBuf[0], data, width, height, stream));
        break;
      }
      case VideoStitch::PixelFormat::YUV422P10: {
        PROPAGATE_FAILURE_STATUS(
            Image::unpackYUV422P10(colorconvBuf[0], colorconvBuf[1], colorconvBuf[2], data, width, height, stream));
        break;
      }
      case VideoStitch::PixelFormat::Grayscale:
        return {Origin::Stitcher, ErrType::UnsupportedAction, "Stitching frames to grayscale writers not implemented"};
      case VideoStitch::PixelFormat::BGRU:
        return {Origin::Stitcher, ErrType::UnsupportedAction, "Stitching frames to BGRU writers not implemented"};
      case VideoStitch::PixelFormat::BGR:
        return {Origin::Stitcher, ErrType::UnsupportedAction, "Stitching frames to BGR writers not implemented"};
      default:
        assert(false);
    }

    // downsample the result
    for (auto it = downsamplers.rbegin(); it != downsamplers.rend(); ++it) {
      PROPAGATE_FAILURE_STATUS(it->second->pushVideo(colorconvBuf, stream));
    }
    return Status::OK();
  }

#define ALLOCATE_IMAGE(img, w, h)                                                               \
  {                                                                                             \
    PotentialValue<GPU::Buffer2D> pot = GPU::Buffer2D::allocate(w, h, "Colorspace conversion"); \
    if (pot.ok()) {                                                                             \
      img.release();                                                                            \
      img = pot.value();                                                                        \
    } else {                                                                                    \
      return pot.status();                                                                      \
    }                                                                                           \
  }

  template <typename Writer>
  Status registerWriter(std::shared_ptr<Writer> writer) {
    // allocate memory for colorspace conversion
    if (colorconvBuf[0].getWidth() == 0) {
      switch (pxFmt) {
        case RGBA:
        case BGRU:
        case F32_C1:
          ALLOCATE_IMAGE(colorconvBuf[0], width * 4, height)
          break;
        case RGB:
        case BGR:
          ALLOCATE_IMAGE(colorconvBuf[0], width * 3, height)
          break;
        case UYVY:
        case YUY2:
        case Grayscale16:
          ALLOCATE_IMAGE(colorconvBuf[0], width * 2, height)
          break;
        case YUV422P10:
          ALLOCATE_IMAGE(colorconvBuf[0], width * 2, height)
          ALLOCATE_IMAGE(colorconvBuf[1], width, height)
          ALLOCATE_IMAGE(colorconvBuf[2], width, height)
          break;
        case YV12:
        case DEPTH:
          ALLOCATE_IMAGE(colorconvBuf[0], width, height)
          ALLOCATE_IMAGE(colorconvBuf[1], width / 2, height / 2)
          ALLOCATE_IMAGE(colorconvBuf[2], width / 2, height / 2)
          break;
        case NV12:
          ALLOCATE_IMAGE(colorconvBuf[0], width, height)
          ALLOCATE_IMAGE(colorconvBuf[1], width, height / 2)
          break;
        default:
          assert(false);
          return {Origin::Stitcher, ErrType::ImplementationError, "Unsupported colorspace for output"};
      }
    }

    // forward the writer to the target downsampler
    const size_t downsamplingRatio = width / writer->getPanoWidth();
    DownsamplingBuffer*& db = downsamplers[downsamplingRatio];
    if (db == nullptr) {
      db = new DownsamplingBuffer(writer->getWidth(), writer->getHeight(), (int)downsamplingRatio,
                                  writer->getPixelFormat());
    }
    return downsamplers[downsamplingRatio]->registerWriter(writer);
  }

  Frame getFrame(AddressSpace addr, size_t ratio) {
    Frame f = downsamplers[ratio]->getFrame(addr);
    f.fmt = pxFmt;
    return f;
  }

 private:
  PixelFormat pxFmt;
  size_t width, height;  // in pixels
  GPU::Buffer2D colorconvBuf[3];
  std::map<size_t, DownsamplingBuffer*> downsamplers;
};

// ---------------------- Frame buffer implementation

template <typename Writer, typename Surface>
Potential<FrameBuffer<Writer, Surface>> FrameBuffer<Writer, Surface>::create(std::shared_ptr<Surface> s) {
  return new FrameBuffer(s);
}

template <typename Writer, typename Surface>
Potential<FrameBuffer<Writer, Surface>> FrameBuffer<Writer, Surface>::create(std::shared_ptr<Surface> s,
                                                                             std::shared_ptr<Writer> writer) {
  Potential<FrameBuffer> ret = create(s);
  if (!ret.ok()) {
    return ret;
  }
  FAIL_RETURN(ret->registerWriter(writer));
  return ret;
}

template <typename Writer, typename Surface>
Potential<FrameBuffer<Writer, Surface>> FrameBuffer<Writer, Surface>::create(
    std::shared_ptr<Surface> s, const std::vector<std::shared_ptr<Writer>>& writers) {
  Potential<FrameBuffer> ret = create(s);
  if (!ret.ok()) {
    return ret;
  }
  for (auto writer : writers) {
    FAIL_RETURN(ret->registerWriter(writer));
  }
  return ret;
}

template <typename Writer, typename Surface>
FrameBuffer<Writer, Surface>::FrameBuffer(std::shared_ptr<Surface> s) : date(-1), surf(s) {}

template <typename Writer, typename Surface>
FrameBuffer<Writer, Surface>::FrameBuffer(FrameBuffer<Writer, Surface>&& other) : date(-1), surf(other.surf) {
  {
    std::unique_lock<std::mutex> lock(other.mutex);
    std::swap(formatBuffers, other.formatBuffers);
  }
}

template <typename Writer, typename Surface>
FrameBuffer<Writer, Surface>& FrameBuffer<Writer, Surface>::operator=(FrameBuffer<Writer, Surface>&& other) {
  if (this != &other) {
    std::unique_lock<std::mutex> lock(mutex);
    std::unique_lock<std::mutex> otherLock(other.mutex);
    std::swap(formatBuffers, other.formatBuffers);
    this->surf = other.surf;
  }
  return *this;
}

template <typename Writer, typename Surface>
FrameBuffer<Writer, Surface>::~FrameBuffer() {
  deleteAllValues(formatBuffers);
}

template <typename Writer, typename Surface>
Frame FrameBuffer<Writer, Surface>::getFrame(PixelFormat fmt, AddressSpace addr, size_t ratio) {
  return formatBuffers[fmt]->getFrame(addr, ratio);
}

template <typename Writer, typename Surface>
Status FrameBuffer<Writer, Surface>::registerWriter(std::shared_ptr<Writer> writer) {
  // This should have been caught in Writer::create.
  assert((int)surf->getWidth() % (int)writer->getPanoWidth() == 0);

  std::unique_lock<std::mutex> lock(mutex);
  // Get or create the format buffer for the writer's format.
  FormatBuffer*& formatBuffer = formatBuffers[writer->getPixelFormat()];
  // const int downsamplingFactor = (int)surf->getWidth() / (int)writer->getPanoWidth();
  if (formatBuffer == nullptr) {
    // TODO: we should be able to simplify things to get rid of paddingTop in FormatBuffer.
    // Use the base padding (without downsampling).
    formatBuffer = new FormatBuffer(writer->getPixelFormat(), surf->getWidth(), surf->getHeight());
  }
  return formatBuffer->registerWriter(writer);
}

// ---------------------- Implementations

Potential<SourceFrameBuffer> SourceFrameBuffer::create(std::shared_ptr<SourceSurface> s) {
  return new SourceFrameBuffer(s);
}

Potential<SourceFrameBuffer> SourceFrameBuffer::create(std::shared_ptr<SourceSurface> s,
                                                       std::shared_ptr<Output::VideoWriter> writer) {
  Potential<SourceFrameBuffer> ret = create(s);
  if (!ret.ok()) {
    return ret;
  }
  FAIL_RETURN(ret->registerWriter(writer));
  return ret;
}

Potential<SourceFrameBuffer> SourceFrameBuffer::create(
    std::shared_ptr<SourceSurface> s, const std::vector<std::shared_ptr<Output::VideoWriter>>& writers) {
  Potential<SourceFrameBuffer> ret = create(s);
  if (!ret.ok()) {
    return ret;
  }
  for (auto writer : writers) {
    FAIL_RETURN(ret->registerWriter(writer));
  }
  return ret;
}

Status SourceFrameBuffer::pushVideo() {
  std::unique_lock<std::mutex> lock(mutex);
  for (auto it = formatBuffers.begin(); it != formatBuffers.end(); ++it) {
    it->second->pushVideo(*surf->pimpl->surface, surf->pimpl->stream);
  }
  surf->pimpl->release();
  return Status::OK();
}

Status SourceFrameBuffer::pushOpenGLVideo() { return Status::OK(); }

template <typename Writer>
Potential<StitchFrameBuffer<Writer>> StitchFrameBuffer<Writer>::create(std::shared_ptr<PanoSurface> s) {
  return new StitchFrameBuffer(s);
}

template <typename Writer>
Potential<StitchFrameBuffer<Writer>> StitchFrameBuffer<Writer>::create(std::shared_ptr<PanoSurface> s,
                                                                       std::shared_ptr<Writer> writer) {
  Potential<StitchFrameBuffer> ret = create(s);
  if (!ret.ok()) {
    return ret;
  }
  FAIL_RETURN(ret->registerWriter(writer));
  return ret;
}

template <typename Writer>
Potential<StitchFrameBuffer<Writer>> StitchFrameBuffer<Writer>::create(
    std::shared_ptr<PanoSurface> s, const std::vector<std::shared_ptr<Writer>>& writers) {
  Potential<StitchFrameBuffer> ret = create(s);
  if (!ret.ok()) {
    return ret;
  }
  for (auto writer : writers) {
    FAIL_RETURN(ret->registerWriter(writer));
  }
  return ret;
}

template <typename Writer>
Status StitchFrameBuffer<Writer>::pushVideo() {
  std::unique_lock<std::mutex> lock(this->mutex);
  this->surf->pimpl->flatten();
  for (auto it = this->formatBuffers.begin(); it != this->formatBuffers.end(); ++it) {
    it->second->pushVideo(this->surf->pimpl->buffer, this->surf->pimpl->stream);
  }
  return Status::OK();
}

template <typename Writer>
Status StitchFrameBuffer<Writer>::pushOpenGLVideo() {
  this->oglSurf->pimpl->acquireWriter();
  this->oglSurf->pimpl->acquire();
  std::unique_lock<std::mutex> lock(this->mutex);
  for (auto it = this->formatBuffers.begin(); it != this->formatBuffers.end(); ++it) {
    it->second->pushVideo(this->oglSurf->pimpl->buffer, this->oglSurf->pimpl->stream);
  }
  this->oglSurf->pimpl->releaseWriter();
  this->oglSurf->pimpl->release();
  return Status::OK();
}

// explicit instantiations
template class FrameBuffer<Output::VideoWriter, SourceSurface>;
template class FrameBuffer<Output::VideoWriter, PanoSurface>;
template class FrameBuffer<Output::StereoWriter, PanoSurface>;
template class StitchFrameBuffer<Output::VideoWriter>;
template class StitchFrameBuffer<Output::StereoWriter>;

}  // namespace Core
}  // namespace VideoStitch
