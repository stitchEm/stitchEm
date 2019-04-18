// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "bufferedReader.hpp"

namespace VideoStitch {
namespace Core {

PotentialValue<Buffer> allocate(size_t size, GPU::HostBuffer<unsigned char>) {
  auto host =
      GPU::HostBuffer<unsigned char>::allocate(size, "Input Frames", GPUHostAllocPinned | GPUHostAllocHostWriteOnly);
  FAIL_RETURN(host.status());
  return Buffer{host.value()};
}

PotentialValue<Buffer> allocate(size_t size, GPU::Buffer<unsigned char>) {
  auto device = GPU::Buffer<unsigned char>::allocate(size, "Input Frames");
  FAIL_RETURN(device.status());
  return Buffer{device.value()};
}

template <typename buffer_t>
PotentialValue<std::vector<Buffer>> allocateBuffers(const Input::VideoReader::Spec& readerSpec, unsigned numBuffers) {
  std::vector<Buffer> buffers;

  auto tryAllocatingBuffers = [&]() -> Status {
    for (unsigned i = 0; i < numBuffers; i++) {
      PotentialValue<Buffer> buf = allocate(readerSpec.frameDataSize, buffer_t());
      FAIL_RETURN(buf.status());
      buffers.push_back(buf.value());
    }
    return Status::OK();
  };

  Status allocationStatus = tryAllocatingBuffers();
  if (!allocationStatus.ok()) {
    for (auto buf : buffers) {
      buf.release();
    }
    return allocationStatus;
  }

  return buffers;
}

BufferedReader::BufferedReader(std::shared_ptr<Input::VideoReader> delegate, std::vector<Buffer> buffers)
    : lastLoadedFrame(), delegate(delegate) {
  lastLoadedFrame.readerStatus = Input::ReadStatus::fromCode<Input::ReadStatusCode::TryAgain>();

  for (auto buf : buffers) {
    // no need to lock, haven't started yet
    availableBuffers.push(buf);
  }

  start();
}

Potential<BufferedReader> BufferedReader::create(std::shared_ptr<Input::VideoReader> reader,
                                                 unsigned preloadCacheSize) {
  FAIL_RETURN(GPU::useDefaultBackendDevice());

  // one buffer for async loading
  // another buffer for the last loaded frame, which needs to be kept for possible reloads and thus is unavailable for
  // preloading
  unsigned minNumBuffers = 2;
  unsigned numBuffers = preloadCacheSize + minNumBuffers;

  switch (reader->getSpec().addressSpace) {
    case Host: {
      auto buffers = allocateBuffers<GPU::HostBuffer<unsigned char>>(reader->getSpec(), numBuffers);
      FAIL_RETURN(buffers.status());
      return new BufferedReader(reader, buffers.value());
    }
    case Device: {
      auto buffers = allocateBuffers<GPU::Buffer<unsigned char>>(reader->getSpec(), numBuffers);
      FAIL_RETURN(buffers.status());
      return new BufferedReader(reader, buffers.value());
    }
  }
  assert(false);
  return Status::OK();
}

BufferedReader::~BufferedReader() {
  {
    std::lock_guard<std::mutex> la(availableMutex);
    stoppingAvailable = true;
  }
  availableCV.notify_all();

  {
    std::lock_guard<std::mutex> ll(loadedMutex);
    stoppingLoaded = true;
  }
  loadedCV.notify_all();

  {
    // make sure any outstanding load have finished
    std::lock_guard<std::mutex> loadingLock(loadedMutex);
    // flush reload frame so it can be released
    updateCurrentFrame({Input::ReadStatus::fromCode<Input::ReadStatusCode::EndOfFile>(), 0, Buffer()});
  }

  // wait for background reading to wind down
  join();

  {
    std::lock_guard<std::mutex> la(availableMutex);
    while (!availableBuffers.empty()) {
      availableBuffers.front().release();
      availableBuffers.pop();
    }
  }

  {
    std::lock_guard<std::mutex> ll(loadedMutex);
    while (!loadedFrames.empty()) {
      auto loaded = std::move(loadedFrames.front());
      loadedFrames.pop();
      loaded.buffer.release();
    }
  }
}

void BufferedReader::run() {
  GPU::useDefaultBackendDevice();

  for (;;) {
    Buffer frame;

    {
      std::unique_lock<std::mutex> lock(availableMutex);
      availableCV.wait(lock, [&]() { return !availableBuffers.empty() || stoppingAvailable; });

      // queue has stopped, wind down reading as well
      if (stoppingAvailable) {
        return;
      }

      frame = availableBuffers.front();
      availableBuffers.pop();
    }

    {
      mtime_t date;
      Input::ReadStatus readStatus;

      std::lock_guard<std::mutex> lock(delegateMutex);
      readStatus = delegate->readFrame(date, frame.rawPtr());

      {
        InputFrame loaded{readStatus, date, frame};

        std::lock_guard<std::mutex> lock(loadedMutex);
        loadedFrames.push(std::move(loaded));
      }
    }
    loadedCV.notify_one();
  }
}

Status BufferedReader::seekFrame(frameid_t seekFrame) {
  // block readFrame
  std::lock_guard<std::mutex> delegateLock(delegateMutex);

  // block loading
  std::lock_guard<std::mutex> loadingLock(loadedMutex);

  if (stoppingLoaded) {
    return Status::OK();
  }

  std::vector<InputFrame> localLoadedFrames;
  while (!loadedFrames.empty()) {
    InputFrame loaded = std::move(loadedFrames.front());
    loadedFrames.pop();
    localLoadedFrames.push_back(std::move(loaded));
  }

  bool seekTargetFrameIsCached = false;

  // TODO API to convert frame ID (used by seek) <--> date (used by readFrame) ?
  mtime_t seekDate =
      (mtime_t)((double)seekFrame * 1000000.0 * (double)getSpec().frameRate.den / (double)getSpec().frameRate.num);

  for (auto& frame : localLoadedFrames) {
    mtime_t frameDate = frame.date;

    if (frameDate == seekDate) {
      seekTargetFrameIsCached = true;
    }

    if (seekTargetFrameIsCached) {
      loadedFrames.push(std::move(frame));
    } else {
      makeBufferAvailable(frame.buffer);
    }
  }

  if (seekTargetFrameIsCached) {
    return Status::OK();
  }

  return delegate->seekFrame(seekFrame);
}

void BufferedReader::makeBufferAvailable(Buffer buf) {
  {
    std::lock_guard<std::mutex> la(availableMutex);
    availableBuffers.push(buf);
  }
  availableCV.notify_one();
}

InputFrame BufferedReader::fetchLoadedFrame() {
  std::unique_lock<std::mutex> lock(loadedMutex);
  loadedCV.wait(lock, [&]() { return !loadedFrames.empty() || stoppingLoaded; });

  // queue has stopped, wind down reading as well
  if (stoppingLoaded) {
    return {Input::ReadStatus::fromCode<Input::ReadStatusCode::EndOfFile>(), -1, Core::Buffer()};
  }

  InputFrame loaded = std::move(loadedFrames.front());
  loadedFrames.pop();
  return loaded;
}

InputFrame BufferedReader::load() {
  InputFrame loadedFrame = fetchLoadedFrame();
  updateCurrentFrame(loadedFrame);
  return loadedFrame;
}

InputFrame BufferedReader::reload() {
  std::lock_guard<std::recursive_mutex> lock(borrowedMutex);
  borrowed[lastLoadedFrame.buffer]++;
  return lastLoadedFrame;
}

void BufferedReader::updateCurrentFrame(InputFrame frame) {
  std::lock_guard<std::recursive_mutex> lock(borrowedMutex);

  InputFrame lastFrame = lastLoadedFrame;
  lastLoadedFrame = frame;
  // 1st borrow for the frame that load() returns
  // 2nd: keep it around for possible reloads
  borrowed[lastLoadedFrame.buffer] = 2;
  releaseBuffer(lastFrame.buffer);
}

void BufferedReader::releaseBuffer(Buffer frame) {
  std::lock_guard<std::recursive_mutex> lock(borrowedMutex);
  if (frame.rawPtr()) {
    borrowed[frame]--;
    if (borrowed[frame] == 0) {
      makeBufferAvailable(frame);
      borrowed.erase(frame);
    }
  }
}

}  // namespace Core
}  // namespace VideoStitch
