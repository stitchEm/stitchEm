// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "frameBuffer.hpp"

#include "libvideostitch/output.hpp"
#include "libvideostitch/frame.hpp"
#include "libvideostitch/stitchOutput.hpp"

#include <map>
#include <mutex>
#include <thread>

namespace VideoStitch {
namespace Core {
/**
 * The internal interface of the stereo stitcher's output:
 * an object holding buffers on the host and device side
 * for a stereoscopic frame.
 */
template <>
class StereoOutput::Pimpl {
 public:
  typedef Output::StereoWriter Writer;

  virtual bool setRenderers(const std::vector<std::shared_ptr<PanoRenderer>>&) = 0;
  virtual bool addRenderer(std::shared_ptr<PanoRenderer>) = 0;
  virtual bool removeRenderer(const std::string&) = 0;
  virtual bool setWriters(const std::vector<std::shared_ptr<Output::StereoWriter>>&) = 0;
  virtual bool addWriter(std::shared_ptr<Output::StereoWriter>) = 0;
  virtual bool removeWriter(const std::string&) = 0;
  virtual bool updateWriter(const std::string&, const Ptv::Value&) = 0;

  /**
   * Returns a pair of device buffer to stitch a panorama to, and an optional stream
   * for asynchronous implementations.
   */
  virtual PanoSurface& acquireLeftFrame(mtime_t) = 0;
  virtual PanoSurface& acquireRightFrame(mtime_t) = 0;

  /**
   * Called by the stitcher when stitching is done.
   */
  virtual Status pushVideo(mtime_t date, Eye eye) = 0;

  virtual ~Pimpl() {}

 protected:
  explicit Pimpl(int64_t width, int64_t height) : width(width), height(height) {}

  int64_t width, height;

 private:
  Pimpl(const Pimpl&) {}
  const Pimpl& operator=(const Pimpl&);
};

/**
 * A function that pushes stereoscopic frames to a bunch of stereoscopic writers.
 */
class StereoWriterPusher {
 public:
  StereoWriterPusher(size_t w, size_t h, const std::vector<std::shared_ptr<Output::StereoWriter>>& writers);
  virtual ~StereoWriterPusher();

  bool setRenderers(const std::vector<std::shared_ptr<PanoRenderer>>&) {
    // XXX TODO FIXME
    return false;
  }
  bool addRenderer(std::shared_ptr<PanoRenderer>) {
    // XXX TODO FIXME
    return false;
  }
  bool removeRenderer(const std::string&) {
    // XXX TODO FIXME
    return false;
  }
  void setCompositor(const std::shared_ptr<GPU::Overlayer>&) {}
  bool setWriters(const std::vector<std::shared_ptr<Output::StereoWriter>>&);
  bool addWriter(std::shared_ptr<Output::StereoWriter>);
  bool removeWriter(const std::string&);
  bool updateWriter(const std::string&, const Ptv::Value&);

 protected:
  void pushVideoToWriters(mtime_t date, std::pair<StereoFrameBuffer*, StereoFrameBuffer*> buffer) const;

 private:
  size_t width;
  std::map<std::string, std::shared_ptr<Output::StereoWriter>> writers;
  std::map<std::string, int> downsamplingFactors;
  mutable std::mutex writersLock;
};

}  // namespace Core
}  // namespace VideoStitch
