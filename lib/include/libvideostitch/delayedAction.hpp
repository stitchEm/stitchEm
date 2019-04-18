// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef DELAYEDACTION_HPP
#define DELAYEDACTION_HPP

#include <atomic>
#include <future>

#include "config.hpp"

namespace VideoStitch {
namespace Core {

class VS_EXPORT DelayedAction {
 public:
  explicit DelayedAction(std::shared_future<void> action);

  void execute();

  bool operator<(const DelayedAction& rhs) const;

 private:
  std::shared_future<void> action;
  size_t order;
};

}  // namespace Core
}  // namespace VideoStitch

#endif  // DELAYEDACTION_HPP
