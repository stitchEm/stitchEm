// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "globalcontroller.hpp"

GlobalController::GlobalController() : impl(nullptr) {}

GlobalController::~GlobalController() {
  Q_ASSERT(impl);
  delete impl;
}

StitcherController* GlobalController::getController() const {
  Q_ASSERT(impl);
  return impl->getController();
}

void GlobalController::createController(int device) {
  Q_ASSERT(impl);
  impl->createController(device);
}

void GlobalController::deleteController() {
  Q_ASSERT(impl);
  impl->deleteController();
}
