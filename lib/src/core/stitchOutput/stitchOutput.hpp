// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "frameBuffer.hpp"

#include "libvideostitch/output.hpp"
#include "libvideostitch/frame.hpp"
#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/utils/semaphore.hpp"

#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace VideoStitch {
namespace Core {

/**
 * The internal interface of the source loopback:
 * an object holding buffers on the host and device side
 * for a video frame.
 */
class ExtractOutput::Pimpl {
 public:
  typedef Output::VideoWriter Writer;

  virtual ~Pimpl() {}

  virtual bool setRenderers(const std::vector<std::shared_ptr<SourceRenderer>>&) = 0;
  virtual bool addRenderer(std::shared_ptr<SourceRenderer>) = 0;
  virtual bool removeRenderer(const std::string&) = 0;
  virtual bool setWriters(const std::vector<std::shared_ptr<Output::VideoWriter>>&) = 0;
  virtual bool addWriter(std::shared_ptr<Output::VideoWriter>) = 0;
  virtual bool removeWriter(const std::string&) = 0;
  virtual bool updateWriter(const std::string&, const Ptv::Value&) = 0;

  videoreaderid_t getSource() const { return sourceIdx; }

  /**
   * Returns a device buffer to extract a source to, and an optional stream
   * for asynchronous implementations.
   */
  virtual GPU::Surface& acquireFrame(mtime_t date, GPU::Stream& stream) = 0;

  /**
   * Called by the stitcher when stitching is done.
   */
  virtual Status pushVideo(mtime_t date) = 0;

 protected:
  explicit Pimpl(int64_t width, int64_t height, videoreaderid_t source = -1)
      : width(width), height(height), sourceIdx(source) {}

  int64_t width, height;
  videoreaderid_t sourceIdx;

 private:
  Pimpl(const Pimpl&);
  const Pimpl& operator=(const Pimpl&);
};

/**
 * The internal interface of the pano stitcher's output:
 * an object holding buffers on the host and device side
 * for a panoramic frame.
 */
template <>
class StitchOutput::Pimpl {
 public:
  typedef Output::VideoWriter Writer;

  virtual ~Pimpl() {}

  virtual bool setRenderers(const std::vector<std::shared_ptr<PanoRenderer>>&) = 0;
  virtual bool addRenderer(std::shared_ptr<PanoRenderer>) = 0;
  virtual bool removeRenderer(const std::string&) = 0;
  virtual void setCompositor(const std::shared_ptr<GPU::Overlayer>&) = 0;
  virtual bool setWriters(const std::vector<std::shared_ptr<Output::VideoWriter>>&) = 0;
  virtual bool addWriter(std::shared_ptr<Output::VideoWriter>) = 0;
  virtual bool removeWriter(const std::string&) = 0;
  virtual bool updateWriter(const std::string&, const Ptv::Value&) = 0;

  /**
   * Returns a device buffer to stitch a panorama to, and an optional stream
   * for asynchronous implementations.
   */
  virtual PanoSurface& acquireFrame(mtime_t) = 0;

  /**
   * Called by the stitcher when stitching is done.
   */
  virtual Status pushVideo(mtime_t) = 0;

 protected:
  explicit Pimpl(int64_t width, int64_t height) : width(width), height(height) {}

  int64_t width, height;

 private:
  Pimpl(const Pimpl&);
  const Pimpl& operator=(const Pimpl&);
};

/**
 * A function that pushes video frames to a bunch of user callbacks.
 */
template <typename FrameBuffer>
class WriterPusher {
 public:
  WriterPusher(size_t w, size_t h, const std::vector<std::shared_ptr<Output::VideoWriter>>& writers);
  virtual ~WriterPusher();

  bool setRenderers(const std::vector<std::shared_ptr<typename FrameBuffer::Renderer>>&);
  bool addRenderer(std::shared_ptr<typename FrameBuffer::Renderer>);
  bool removeRenderer(const std::string&);
  void setCompositor(const std::shared_ptr<GPU::Overlayer>&);
  bool setWriters(const std::vector<std::shared_ptr<Output::VideoWriter>>&);
  bool addWriter(std::shared_ptr<Output::VideoWriter>);
  bool removeWriter(const std::string&);
  bool updateWriter(const std::string&, const Ptv::Value&);

 protected:
  void pushVideoToWriters(mtime_t date, FrameBuffer* delegate) const;

 private:
  size_t width;
  std::map<std::string, std::shared_ptr<typename FrameBuffer::Renderer>> renderers;
  mutable std::mutex renderersLock;
  std::map<std::string, std::shared_ptr<Output::VideoWriter>> writers;
  mutable std::mutex writersLock;
  std::map<std::string, int> downsamplingFactors;
  std::shared_ptr<GPU::Overlayer> compositor;
  mutable std::mutex compositorLock;
};

typedef WriterPusher<PanoFrameBuffer> PanoWriterPusher;
typedef WriterPusher<SourceFrameBuffer> SourceWriterPusher;

}  // namespace Core
}  // namespace VideoStitch
