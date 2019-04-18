// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <libvideostitch/algorithm.hpp>
#include <atomic>

class StitcherController;

class VS_GUI_EXPORT BackendInitializerProgressReporter : public VideoStitch::Util::Algorithm::ProgressReporter {
 public:
  explicit BackendInitializerProgressReporter(StitcherController* stitcherController);

  bool notify(const std::string& message, double percent) override;
  void tryToCancel();
  void reset();

 private:
  StitcherController* controller = nullptr;
  std::atomic<bool> cancel;
};
