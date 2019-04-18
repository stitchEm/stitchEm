// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/input.hpp"

#include <cassert>
#include <map>
#include <mutex>

namespace VideoStitch {
namespace Input {
/**
 * A reader that holds some GPU state. Use if you need some thread-local GPU buffers.
 *
 * NOTE: It's a template, but the implementation needs to be hidden since it uses some C++11 features that CUDA does not
 * support yet. Whenever you use it with a new StateT, please instanciate explicitly in the CC file.
 */
template <typename StateT>
class StatefulReader : public VideoReader {
 public:
  typedef StatefulReader<StateT> StatefulBase;  ///< StatefulReader base class

  /** StatefulReader constructor. See Reader documentation for description of parameters. */
  StatefulReader(int64_t width, int64_t height, int64_t frameDataSize, PixelFormat format, AddressSpace addr,
                 FrameRate frameRate, int firstFrame, int lastFrame, bool isProcedural,
                 const unsigned char* maskHostBuffer, int flags = 0);

  virtual ~StatefulReader();

  /**
   * Returns the data associated with the current device/thread, or NULL if there is no data.
   */
  const StateT* getCurrentDeviceData() const;

  /**
   * Sets the data associated with the current device/thread. Returns true in success.
   */
  Status setCurrentDeviceData(const StateT& state);

  Status perThreadInit() {
    // This does not make sense.
    return {Origin::Input, ErrType::ImplementationError, "Cannot init stateful reader"};
  }

  // Please base-call if overridden.
  void perThreadCleanup();

 private:
  /**
   * Returns the current device/thread, or -1 on error.
   */
  int getCurrentDevice() const;

  mutable std::mutex mutex;
  std::map<int, StateT> perDeviceData;
};
}  // namespace Input
}  // namespace VideoStitch
