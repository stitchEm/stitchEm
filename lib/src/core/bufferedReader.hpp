// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/input.hpp"

#include "buffer.hpp"
#include "common/thread.hpp"
#include "gpu/hostBuffer.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <unordered_map>

namespace std {

template <>
struct hash<VideoStitch::Core::Buffer> {
  std::size_t operator()(const VideoStitch::Core::Buffer& k) const { return hash<unsigned char*>()(k.rawPtr()); }
};
}  // namespace std

namespace VideoStitch {
namespace Core {

struct InputFrame {
  Input::ReadStatus readerStatus;
  mtime_t date;
  Buffer buffer;

  bool operator==(const InputFrame& other) const {
    return readerStatus.getCode() == other.readerStatus.getCode() && date == other.date && buffer == other.buffer;
  }
};

class BufferedReader : public Thread {
 public:
  static Potential<BufferedReader> create(std::shared_ptr<Input::VideoReader> delegate, unsigned preloadCacheSize);
  ~BufferedReader();

  Status seekFrame(frameid_t date);

  InputFrame load();

  InputFrame reload();

  void releaseBuffer(Buffer frame);

  // TODO remove
  // keep references to reader controller in the stitcher, not the readers
  std::shared_ptr<Input::VideoReader> getDelegate() { return delegate; }

  const Input::VideoReader::Spec& getSpec() const { return delegate->getSpec(); }

  /**
   * Returns the first frame in the sequence (inclusive).
   */
  frameid_t getFirstFrame() const { return delegate->getFirstFrame(); }

  /**
   * Returns the last frame in the sequence (inclusive), or NO_LAST_FRAME.
   */
  frameid_t getLastFrame() const { return delegate->getLastFrame(); }

  Status perThreadInit() { return delegate->perThreadInit(); }

  void perThreadCleanup() { return delegate->perThreadCleanup(); }

  virtual void run();

 private:
  BufferedReader(std::shared_ptr<Input::VideoReader> delegate, std::vector<Buffer> buffers);

  void updateCurrentFrame(InputFrame frame);
  void makeBufferAvailable(Buffer buf);
  InputFrame fetchLoadedFrame();

  std::recursive_mutex borrowedMutex;
  InputFrame lastLoadedFrame;
  std::unordered_map<Buffer, int> borrowed;

  std::mutex availableMutex;
  std::condition_variable availableCV;
  std::queue<Buffer> availableBuffers;
  bool stoppingAvailable = false;

  std::mutex loadedMutex;
  std::condition_variable loadedCV;
  std::queue<InputFrame> loadedFrames;
  bool stoppingLoaded = false;

  std::mutex delegateMutex;
  std::shared_ptr<Input::VideoReader> delegate;
};

}  // namespace Core
}  // namespace VideoStitch
