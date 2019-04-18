// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/delayedAction.hpp"

namespace VideoStitch {
namespace Core {

static std::atomic_size_t orderCounter(0);

DelayedAction::DelayedAction(std::shared_future<void> action) : action(action), order(orderCounter++) {}

void DelayedAction::execute() { action.get(); }

bool DelayedAction::operator<(const DelayedAction& rhs) const { return this->order < rhs.order; }

}  // namespace Core
}  // namespace VideoStitch
