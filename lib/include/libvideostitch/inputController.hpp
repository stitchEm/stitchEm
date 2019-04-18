// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "input.hpp"

#include <vector>

namespace VideoStitch {

namespace Core {

enum class ControllerStatusCode {
  // mandatory status
  Ok,
  ErrorWithStatus,
  // custom stitcher state
  // end of file reached
  EndOfStream
};

typedef Result<ControllerStatusCode> ControllerStatus;

class VS_EXPORT InputController {
 public:
  /**
   * Returns the maximum of the readers' first frame numbers.
   * This is the first frame that can be seeked and stitched.
   */
  virtual frameid_t getFirstReadableFrame() const = 0;

  /**
   * Returns the minimum of the readers' last frame numbers, or NO_LAST_FRAME if unbounded.
   */
  virtual frameid_t getLastReadableFrame() const = 0;

  /**
   * Returns the last frame that can be seeked and stitched, or NO_LAST_FRAME if unbounded.
   * This is not necessarily the same as getLastReadableFrame() because of frame offsets.
   */
  virtual frameid_t getLastStitchableFrame() const = 0;

  /**
   * Returns the reader last frames
   */
  virtual std::vector<frameid_t> getLastFrames() const = 0;

  /**
   * Returns the frame rate of the controller's inputs.
   */
  virtual FrameRate getFrameRate() const = 0;

  /**
   * Reader Spec accessor.
   * @returns The i-th reader spec.
   */
  virtual const Input::VideoReader::Spec& getReaderSpec(int i) const = 0;
};

}  // namespace Core
}  // namespace VideoStitch
