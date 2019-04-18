// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "util.hpp"

#include "libvideostitch/logging.hpp"

namespace VideoStitch {

namespace Util {

void TimeoutHandler::logTimeout() {
  Logger::get(Logger::Warning) << "[TimeoutHandler] Operation timed out, interrupting!" << std::endl;
}

}  // namespace Util
}  // namespace VideoStitch
