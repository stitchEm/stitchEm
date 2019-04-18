// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/inputController.hpp"

#include "readerController.hpp"

#include <vector>

namespace VideoStitch {

namespace Core {

class VS_EXPORT InputControllerImpl : public virtual InputController {
 public:
  explicit InputControllerImpl(ReaderController *ctrl) : readerController(ctrl) {}

  virtual ~InputControllerImpl() {
    readerController->cleanReaders();
    delete readerController;
  }

  frameid_t getFirstReadableFrame() const final override { return readerController->getFirstReadableFrame(); }

  frameid_t getLastReadableFrame() const final override { return readerController->getLastReadableFrame(); }

  frameid_t getLastStitchableFrame() const final override { return readerController->getLastStitchableFrame(); }

  std::vector<frameid_t> getLastFrames() const final override { return readerController->getLastFrames(); }

  FrameRate getFrameRate() const final override { return readerController->getFrameRate(); }

  const Input::VideoReader::Spec &getReaderSpec(int i) const final override {
    return readerController->getReaderSpec(i);
  }

 protected:
  ReaderController *readerController;
};

}  // namespace Core
}  // namespace VideoStitch
