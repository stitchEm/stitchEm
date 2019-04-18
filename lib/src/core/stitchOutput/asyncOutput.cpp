// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "asyncOutput.hpp"

#include "common/container.hpp"

#include "libvideostitch/gpu_device.hpp"

namespace VideoStitch {
namespace Core {
/**
 * Async*Buffer
 */
Potential<AsyncSourceBuffer> AsyncSourceBuffer::create(
    const std::vector<std::shared_ptr<SourceFrameBuffer::Surface>>& surfs,
    const std::vector<std::shared_ptr<Output::VideoWriter>>& writers) {
  auto ab = std::make_unique<AsyncSourceBuffer>();
  FAIL_RETURN(ab->initialize(surfs, writers));
  return ab.release();
}

Potential<AsyncPanoBuffer> AsyncPanoBuffer::create(const std::vector<std::shared_ptr<PanoFrameBuffer::Surface>>& surfs,
                                                   const std::vector<std::shared_ptr<Output::VideoWriter>>& writers) {
  auto ab = std::make_unique<AsyncPanoBuffer>();
  FAIL_RETURN(ab->initialize(surfs, writers));
  return ab.release();
}

template <typename FrameBuffer>
Status AsyncBuffer<FrameBuffer>::initialize(const std::vector<std::shared_ptr<typename FrameBuffer::Surface>>& surfs,
                                            const std::vector<std::shared_ptr<Output::VideoWriter>>& writers) {
  FAIL_RETURN(GPU::useDefaultBackendDevice());
  for (auto surf : surfs) {
    Potential<FrameBuffer> frame = FrameBuffer::create(surf, writers);
    if (!frame.ok()) {
      return frame.status();
    }
    blankFrames.push_back(frame.object());
    allFrames.push_back(frame.release());
  }

  return Status::OK();
}

template <typename FrameBuffer>
AsyncBuffer<FrameBuffer>::~AsyncBuffer() {
  deleteAll(allFrames);
}

// -------- Stitcher thread functions

GPU::Surface& AsyncSourceBuffer::acquireFrame(mtime_t date, GPU::Stream& stream) {
  SourceFrameBuffer* frame = getCurrentFrame(date);
  return frame->acquireFrame(date, stream);
}

PanoSurface& AsyncPanoBuffer::acquireFrame(mtime_t date) {
  PanoFrameBuffer* frame = getCurrentFrame(date);
  return frame->acquireFrame(date);
}

template <typename FrameBuffer>
Status AsyncBuffer<FrameBuffer>::pushVideo(mtime_t date) {
  FrameBuffer* frame = getCurrentFrame(date);
  // release the surf frame
  frame->releaseFrame();
  inUse.erase(date);

  // put the frame as "stitched".
  {
    std::unique_lock<std::mutex> lock(stMu);
    stitchedFrames.push_back(std::make_pair(date, frame));
  }
  stCond.notify_one();

  return Status::OK();
}

template <typename FrameBuffer>
auto AsyncBuffer<FrameBuffer>::getCurrentFrame(mtime_t date) -> FrameBuffer* {
  // first of all have we already seen this frame ?
  FrameBuffer* frame = inUse[date];

  if (frame == nullptr) {
    // new frame, wait for a blank frame to be available
    {
      std::unique_lock<std::mutex> lock(bkMu);
      bkCond.wait(lock, [this] { return blankFrames.size() > 0; });
      inUse[date] = blankFrames.front();
      blankFrames.pop_front();
    }
  }
  return inUse[date];
}

template <typename FrameBuffer>
auto AsyncBuffer<FrameBuffer>::getUsedFrame(mtime_t date) -> FrameBuffer* {
  return inUse.at(date);
}

// -------- Register thread functions

template <typename FrameBuffer>
Status AsyncBuffer<FrameBuffer>::registerWriters(const std::vector<std::shared_ptr<Output::VideoWriter>>& writers) {
  for (auto writer : writers) {
    FAIL_RETURN(registerWriter(writer));
  }
  return Status::OK();
}

template <typename FrameBuffer>
Status AsyncBuffer<FrameBuffer>::registerWriter(std::shared_ptr<Output::VideoWriter> writer) {
  FAIL_RETURN(GPU::useDefaultBackendDevice());
  for (auto frame : allFrames) {
    FAIL_RETURN(frame->registerWriter(writer));
  }
  return Status::OK();
}

template class AsyncBuffer<PanoFrameBuffer>;
template class AsyncBuffer<SourceFrameBuffer>;

/**
 * AsyncBufferedOutput
 */
template <typename Pimpl, typename AsyncBuffer, typename Pusher, typename Device>
bool AsyncBufferedOutput<Pimpl, AsyncBuffer, Pusher, Device>::setRenderers(
    const std::vector<std::shared_ptr<typename AsyncBuffer::FB::Renderer>>& r) {
  return Pusher::setRenderers(r);
}

template <typename Pimpl, typename AsyncBuffer, typename Pusher, typename Device>
AsyncBufferedOutput<Pimpl, AsyncBuffer, Pusher, Device>::AsyncBufferedOutput(
    const std::vector<std::shared_ptr<typename AsyncBuffer::FB::Surface>>& surfs,
    const std::vector<std::shared_ptr<Writer>>& writers)
    : Pimpl(surfs[0]->getWidth(), surfs[0]->getHeight()),
      Pusher(surfs[0]->getWidth(), surfs[0]->getHeight(), writers),
      shutdown(false) {}

template <typename Pimpl, typename AsyncBuffer, typename Pusher, typename Device>
Status AsyncBufferedOutput<Pimpl, AsyncBuffer, Pusher, Device>::initialize(
    const std::vector<std::shared_ptr<typename AsyncBuffer::FB::Surface>>& surfs,
    const std::vector<std::shared_ptr<Writer>>& writers) {
  FAIL_RETURN(AsyncBuffer::initialize(surfs, writers));
  worker = new std::thread(consumerThread, this);
  return Status::OK();
}

template <typename Pimpl, typename AsyncBuffer, typename Pusher, typename Device>
AsyncBufferedOutput<Pimpl, AsyncBuffer, Pusher, Device>::~AsyncBufferedOutput() {
  shutdown = true;
  this->stCond.notify_one();

  if (worker != nullptr) {
    worker->join();
    delete worker;
    worker = nullptr;
  }
}

template <typename Pimpl, typename AsyncBuffer, typename Pusher, typename Device>
bool AsyncBufferedOutput<Pimpl, AsyncBuffer, Pusher, Device>::setWriters(
    const std::vector<std::shared_ptr<Writer>>& writers) {
  AsyncBuffer::registerWriters(writers);
  return Pusher::setWriters(writers);
}

template <typename Pimpl, typename AsyncBuffer, typename Pusher, typename Device>
bool AsyncBufferedOutput<Pimpl, AsyncBuffer, Pusher, Device>::addWriter(std::shared_ptr<Writer> writer) {
  // Be careful to always register the writer before adding it
  AsyncBuffer::registerWriter(writer);
  return Pusher::addWriter(writer);
}

template <typename Pimpl, typename AsyncBuffer, typename Pusher, typename Device>
bool AsyncBufferedOutput<Pimpl, AsyncBuffer, Pusher, Device>::removeWriter(const std::string& id) {
  return Pusher::removeWriter(id);
}

template <typename Pimpl, typename AsyncBuffer, typename Pusher, typename Device>
bool AsyncBufferedOutput<Pimpl, AsyncBuffer, Pusher, Device>::updateWriter(const std::string& id,
                                                                           const Ptv::Value& config) {
  return Pusher::updateWriter(id, config);
}

template <typename Pimpl, typename AsyncBuffer, typename Pusher, typename Device>
void AsyncBufferedOutput<Pimpl, AsyncBuffer, Pusher, Device>::consumerThread(AsyncBufferedOutput* that) {
  if (!GPU::useDefaultBackendDevice().ok()) {
    return;
  }
  std::pair<mtime_t, typename AsyncBuffer::Frame> frame;
  for (;;) {
    // Wait for a frame to be scheduled.
    {
      std::unique_lock<std::mutex> lock(that->stMu);
      that->stCond.wait(lock, [that] { return that->stitchedFrames.size() > 0 || that->shutdown; });
      if (that->shutdown && that->stitchedFrames.size() == 0) {
        return;
      }
      frame = that->stitchedFrames.front();
      that->stitchedFrames.pop_front();
    }

    // Wait for the scheduled frame to be produced.
    that->synchronize(frame.second);

    // Consume it
    // Util::SimpleProfiler prof("  write frame", false, Logger::get(Logger::Verbose));
    that->pushVideoToWriters(frame.first, frame.second);

    // Put the frame back as blank
    {
      std::unique_lock<std::mutex> lock(that->bkMu);
      that->blankFrames.push_back(frame.second);
    }
    that->bkCond.notify_all();
  }
}

// ------------- Stereoscopy --------------------

/**
 * AsyncStereoBuffer
 */
Potential<AsyncStereoBuffer> AsyncStereoBuffer::create(
    const std::vector<std::shared_ptr<PanoSurface>>& surfs,
    const std::vector<std::shared_ptr<Output::StereoWriter>>& writers) {
  std::unique_ptr<AsyncStereoBuffer> ret(new AsyncStereoBuffer);
  FAIL_RETURN(ret->initialize(surfs, writers));
  return ret.release();
}

Status AsyncStereoBuffer::initialize(const std::vector<std::shared_ptr<PanoSurface>>& surfs,
                                     const std::vector<std::shared_ptr<Output::StereoWriter>>& writers) {
  FAIL_RETURN(GPU::useDefaultBackendDevice());
  for (auto surf : surfs) {
    Potential<StereoFrameBuffer> left = StereoFrameBuffer::create(surf, writers);
    if (!left.ok()) {
      return left.status();
    }

    Potential<StereoFrameBuffer> right = StereoFrameBuffer::create(surf, writers);
    if (!right.ok()) {
      return right.status();
    }

    allFrames.push_back(std::make_pair(left.object(), right.object()));
    blankFrames.push_back(std::make_pair(left.release(), right.release()));
  }

  return Status::OK();
}

AsyncStereoBuffer::~AsyncStereoBuffer() {
  for (auto stereoFrame : allFrames) {
    // delete left buffer
    delete stereoFrame.first;
    // delete right buffer
    delete stereoFrame.second;
  }
}

PanoSurface& AsyncStereoBuffer::acquireLeftFrame(mtime_t date) { return acquireFrame(date, LeftEye); }

PanoSurface& AsyncStereoBuffer::acquireRightFrame(mtime_t date) { return acquireFrame(date, RightEye); }

PanoSurface& AsyncStereoBuffer::acquireFrame(mtime_t date, Eye eye) {
  Frame frame = getCurrentFrame(date);
  if (eye == LeftEye) {
    return frame.first->acquireFrame(date);
  } else {
    return frame.second->acquireFrame(date);
  }
}

Status AsyncStereoBuffer::pushVideo(mtime_t date, Eye eye) {
  Frame stereoFrame = getCurrentFrame(date);
  Status res;
  if (eye == LeftEye) {
    res = stereoFrame.first->pushVideo();
  } else {
    res = stereoFrame.second->pushVideo();
  }

  // prevent concurrent access from the two stitcher's threads (left/right)
  // by checking inUse again under lock
  {
    std::unique_lock<std::mutex> lock(stMu);
    if (inUse.find(date) != inUse.end()) {
      inUse.erase(date);
      stitchedFrames.push_back(std::make_pair(date, stereoFrame));
      stCond.notify_one();
    }
  }

  return res;
}

auto AsyncStereoBuffer::getCurrentFrame(mtime_t date) -> Frame {
  // if new frame, wait for a blank frame to be available
  std::unique_lock<std::mutex> bkLock(bkMu);
  bkCond.wait(bkLock, [this, date] { return blankFrames.size() > 0 || inUse.find(date) != inUse.end(); });
  // here we need to check again for inUse : indeed, the left & right stitchers
  // could have been waiting simultaneously on a blank frame to be available
  if (inUse.find(date) == inUse.end()) {
    inUse[date] = blankFrames.front();
    blankFrames.pop_front();
  }
  return inUse[date];
}

auto AsyncStereoBuffer::getUsedFrame(mtime_t date) -> Frame { return inUse.at(date); }

void AsyncStereoBuffer::synchronize(Frame frame) {
  frame.first->streamSynchronize();
  frame.second->streamSynchronize();
}

Status AsyncStereoBuffer::registerWriters(const std::vector<std::shared_ptr<Output::StereoWriter>>& writers) {
  for (auto writer : writers) {
    FAIL_RETURN(registerWriter(writer));
  }
  return Status::OK();
}

Status AsyncStereoBuffer::registerWriter(std::shared_ptr<Output::StereoWriter> writer) {
  for (auto frame : allFrames) {
    FAIL_RETURN(frame.first->registerWriter(writer));
    FAIL_RETURN(frame.second->registerWriter(writer));
  }
  return Status::OK();
}

template class AsyncBufferedOutput<ExtractOutput::Pimpl, AsyncSourceBuffer, SourceWriterPusher, PanoDeviceDefinition>;
template class AsyncBufferedOutput<StitchOutput::Pimpl, AsyncPanoBuffer, PanoWriterPusher, PanoDeviceDefinition>;
template class AsyncBufferedOutput<StereoOutput::Pimpl, AsyncStereoBuffer, StereoWriterPusher, StereoDeviceDefinition>;
}  // namespace Core
}  // namespace VideoStitch
