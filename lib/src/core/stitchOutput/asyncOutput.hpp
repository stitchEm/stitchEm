// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "frameBuffer.hpp"
#include "stitchOutput.hpp"
#include "stereoOutput.hpp"

#include "libvideostitch/controller.hpp"

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace VideoStitch {
namespace Output {
class VideoWriter;
class StereoWriter;
}  // namespace Output
namespace Core {
/**
 * A set of host panoramic buffers that can be filled asynchronously.
 *
 * It buffers N frames and waits when no frame is available.
 * Warning: on destruction, waits for all pending frames to be written.
 * This can effectively block forever if there are missing frames.
 *
 * Frames should start at 0.
 *
 * Past frames will be ignored, current frame undergoes a
 * special treatment if refilled (for restitch).
 * Future frames are buffered when possible, if not fill()
 * becomes blocking.
 *
 * 3 threads are manipulating the panoramic buffers.
 * - "register callbacks" thread : read all buffers, don't change their state (stitched / blank / in use),
 *   modify them
 * - "stitcher" thread : pop one from blanks, put one in stitched
 * - "consumer" thread : pop one from stitched, put one in blank
 */
template <typename FrameBuffer>
class AsyncBuffer {
 public:
  typedef FrameBuffer* Frame;
  typedef FrameBuffer FB;

  Status initialize(const std::vector<std::shared_ptr<typename FrameBuffer::Surface>>&,
                    const std::vector<std::shared_ptr<Output::VideoWriter>>& writers);

  virtual ~AsyncBuffer();

  Status pushVideo(mtime_t date);

  Status registerWriters(const std::vector<std::shared_ptr<Output::VideoWriter>>&);
  Status registerWriter(std::shared_ptr<Output::VideoWriter>);

 protected:
  void synchronize(FrameBuffer* frame) { frame->streamSynchronize(); }
  FrameBuffer* getUsedFrame(mtime_t date);
  FrameBuffer* getCurrentFrame(mtime_t date);

  // Frames
  std::mutex bkMu;
  std::condition_variable bkCond;
  std::deque<FrameBuffer*> blankFrames;

  std::mutex stMu;
  std::condition_variable stCond;

  std::deque<std::pair<mtime_t, FrameBuffer*>> stitchedFrames;
  bool shutDown = false;

  // Hold the frames while the stitcher schedules everything
  // The stitcher asks first for the device buffer in which
  // to schedule the stitching, then, once everything is scheduled,
  // it asks for the host buffer where to render the results.
  //
  // Since multiple frames can be scheduled at the same time, if more
  // than double buffering is used, we ask
  // to store the correspondence in between.
  // Unlocked, since only the "stitcher" thread
  // accesses it.
  std::map<mtime_t, FrameBuffer*> inUse;

  // used by the "register" thread only
  std::vector<FrameBuffer*> allFrames;
};

class AsyncSourceBuffer : public AsyncBuffer<SourceFrameBuffer> {
 public:
  static Potential<AsyncSourceBuffer> create(const std::vector<std::shared_ptr<SourceFrameBuffer::Surface>>&,
                                             const std::vector<std::shared_ptr<Output::VideoWriter>>&);

  GPU::Surface& acquireFrame(mtime_t date, GPU::Stream& stream);
};

class AsyncPanoBuffer : public AsyncBuffer<PanoFrameBuffer> {
 public:
  static Potential<AsyncPanoBuffer> create(const std::vector<std::shared_ptr<PanoFrameBuffer::Surface>>&,
                                           const std::vector<std::shared_ptr<Output::VideoWriter>>&);

  PanoSurface& acquireFrame(mtime_t date);
};

/**
 * A set of host stereoscopic buffers that can be filled asynchronously.
 *
 * It buffers N frames and waits when no frame is available.
 * Warning: on destruction, waits for all pending frames to be written.
 * This can effectively block forever if there are missing frames.
 *
 * Frames should start at 0.
 *
 * Past frames will be ignored, current frame undergoes a
 * special treatment if refilled (for restitch).
 * Future frames are buffered when possible, if not fill()
 * becomes blocking.
 * 3 types of threads are manipulating the panoramic buffers.
 * - "register callbacks" thread : read all buffers, don't change their state (stitched / blank / in use),
 *   modify them
 * - 2 "stitcher" threads : pop one from blanks, put one in stitched, collaborate (one pop - one push)
 * - "consumer" thread : pop one from stitched, put one in blank
 */
class AsyncStereoBuffer {
 public:
  typedef std::pair<StereoFrameBuffer*, StereoFrameBuffer*> Frame;
  typedef PanoFrameBuffer FB;

  static Potential<AsyncStereoBuffer> create(const std::vector<std::shared_ptr<PanoSurface>>&,
                                             const std::vector<std::shared_ptr<Output::StereoWriter>>& writers);
  Status initialize(const std::vector<std::shared_ptr<PanoSurface>>&,
                    const std::vector<std::shared_ptr<Output::StereoWriter>>& writers);

  virtual ~AsyncStereoBuffer();

  PanoSurface& acquireLeftFrame(mtime_t);
  PanoSurface& acquireRightFrame(mtime_t);

  Status pushVideo(mtime_t date, Eye eye);

  Status registerWriters(const std::vector<std::shared_ptr<Output::StereoWriter>>&);
  Status registerWriter(std::shared_ptr<Output::StereoWriter>);

 protected:
  AsyncStereoBuffer() {}

  PanoSurface& acquireFrame(mtime_t date, Eye eye);
  void synchronize(Frame frame);
  Frame getUsedFrame(mtime_t date);
  Frame getCurrentFrame(mtime_t date);

  // Frames
  std::mutex bkMu;
  std::condition_variable bkCond;
  std::deque<Frame> blankFrames;

  std::mutex stMu;
  std::condition_variable stCond;
  std::deque<std::pair<mtime_t, Frame>> stitchedFrames;
  bool shutDown = false;

  // Hold the frames while the stitcher schedules everything
  // The stitcher asks first for the device buffer in which
  // to schedule the stitching, then, once everything is scheduled,
  // it asks for the host buffer where to render the results.
  //
  // Since multiple frames can be scheduled at the same time, if more
  // than double buffering is used, we ask
  // to store the correspondence in between.
  std::map<mtime_t, Frame> inUse;

  // used by the "register" thread only
  std::vector<Frame> allFrames;
};

template <typename Pimpl, typename AsyncBuffer, typename Pusher, typename Device>
class AsyncBufferedOutput : public Pimpl, protected AsyncBuffer, protected Pusher {
 public:
  typedef typename Pimpl::Writer Writer;

  AsyncBufferedOutput(const std::vector<std::shared_ptr<typename AsyncBuffer::FB::Surface>>&,
                      const std::vector<std::shared_ptr<Writer>>& writers);
  virtual ~AsyncBufferedOutput();

  virtual bool setRenderers(const std::vector<std::shared_ptr<typename AsyncBuffer::FB::Renderer>>&) override;
  virtual bool addRenderer(std::shared_ptr<typename AsyncBuffer::FB::Renderer> renderer) override {
    return Pusher::addRenderer(renderer);
  }
  virtual bool removeRenderer(const std::string& name) override { return Pusher::removeRenderer(name); }
  virtual bool setWriters(const std::vector<std::shared_ptr<Writer>>&) override;
  virtual bool addWriter(std::shared_ptr<Writer>) override;
  virtual bool removeWriter(const std::string&) override;
  virtual bool updateWriter(const std::string&, const Ptv::Value&) override;

 protected:
  Status initialize(const std::vector<std::shared_ptr<typename AsyncBuffer::FB::Surface>>&,
                    const std::vector<std::shared_ptr<Writer>>& writers);

 private:
  static void consumerThread(AsyncBufferedOutput* that);

  std::thread* worker;
  std::atomic<bool> shutdown;
};

class AsyncSourceOutput : public AsyncBufferedOutput<ExtractOutput::Pimpl, AsyncSourceBuffer,
                                                     WriterPusher<SourceFrameBuffer>, PanoDeviceDefinition> {
 public:
  typedef AsyncBufferedOutput<ExtractOutput::Pimpl, AsyncSourceBuffer, WriterPusher<SourceFrameBuffer>,
                              PanoDeviceDefinition>
      Base;

  static Potential<AsyncSourceOutput> create(const std::vector<std::shared_ptr<SourceSurface>>& surfs,
                                             const std::vector<std::shared_ptr<SourceRenderer>>& renderers,
                                             const std::vector<std::shared_ptr<Output::VideoWriter>>& writers,
                                             int source) {
    AsyncSourceOutput* aso = new AsyncSourceOutput(surfs, renderers, writers, source);
    FAIL_RETURN(aso->initialize(surfs, writers));
    return aso;
  }

  Status pushVideo(mtime_t date) override { return AsyncBuffer<SourceFrameBuffer>::pushVideo(date); }

  GPU::Surface& acquireFrame(mtime_t date, GPU::Stream& stream) override {
    return AsyncSourceBuffer::acquireFrame(date, stream);
  }

 private:
  AsyncSourceOutput(const std::vector<std::shared_ptr<SourceSurface>>& surfs,
                    const std::vector<std::shared_ptr<SourceRenderer>>& renderers,
                    const std::vector<std::shared_ptr<Output::VideoWriter>>& writers, int source)
      : Base(surfs, writers) {
    sourceIdx = source;
    setRenderers(renderers);
  }
};

class AsyncStitchOutput : public AsyncBufferedOutput<StitchOutput::Pimpl, AsyncPanoBuffer,
                                                     WriterPusher<PanoFrameBuffer>, PanoDeviceDefinition> {
 public:
  static Potential<AsyncStitchOutput> create(const std::vector<std::shared_ptr<PanoSurface>>& surfs,
                                             const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                                             const std::vector<std::shared_ptr<Output::VideoWriter>>& writers) {
    AsyncStitchOutput* aso = new AsyncStitchOutput(surfs, renderers, writers);
    FAIL_RETURN(aso->initialize(surfs, writers));
    return aso;
  }

  Status pushVideo(mtime_t date) override { return AsyncBuffer<PanoFrameBuffer>::pushVideo(date); }

  virtual void setCompositor(const std::shared_ptr<GPU::Overlayer>& c) override {
    WriterPusher<PanoFrameBuffer>::setCompositor(c);
  }

  PanoSurface& acquireFrame(mtime_t date) override { return AsyncPanoBuffer::acquireFrame(date); }

 protected:
  typedef AsyncBufferedOutput<StitchOutput::Pimpl, AsyncPanoBuffer, WriterPusher<PanoFrameBuffer>, PanoDeviceDefinition>
      Base;

  AsyncStitchOutput(const std::vector<std::shared_ptr<PanoSurface>>& surfs,
                    const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                    const std::vector<std::shared_ptr<Output::VideoWriter>>& writers)
      : Base(surfs, writers) {
    setRenderers(renderers);
  }
};

class AsyncStereoOutput
    : public AsyncBufferedOutput<StereoOutput::Pimpl, AsyncStereoBuffer, StereoWriterPusher, StereoDeviceDefinition> {
 public:
  static Potential<AsyncStereoOutput> create(const std::vector<std::shared_ptr<PanoSurface>>& surfs,
                                             const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                                             const std::vector<std::shared_ptr<Output::StereoWriter>>& writers) {
    AsyncStereoOutput* ret = new AsyncStereoOutput(surfs, renderers, writers);
    FAIL_RETURN(ret->initialize(surfs, writers));
    return ret;
  }
  virtual ~AsyncStereoOutput() {}

  Status pushVideo(mtime_t date, Eye eye) override { return AsyncStereoBuffer::pushVideo(date, eye); }

  virtual PanoSurface& acquireLeftFrame(mtime_t date) override { return AsyncStereoBuffer::acquireLeftFrame(date); }
  virtual PanoSurface& acquireRightFrame(mtime_t date) override { return AsyncStereoBuffer::acquireRightFrame(date); }

 protected:
  AsyncStereoOutput(const std::vector<std::shared_ptr<PanoSurface>>& surfs,
                    const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                    const std::vector<std::shared_ptr<Output::StereoWriter>>& writers)
      : AsyncBufferedOutput(surfs, writers) {
    setRenderers(renderers);
  }
};
}  // namespace Core
}  // namespace VideoStitch
