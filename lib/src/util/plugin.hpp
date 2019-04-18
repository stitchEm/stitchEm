// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/plugin.hpp"
#include "libvideostitch/logging.hpp"

#include <mutex>

namespace VideoStitch {

namespace Plugin {

extern std::mutex pluginsMutex;  // TODO: use a read/write mutex.
}
}  // namespace VideoStitch
