// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "flowSequence.hpp"

#include "backend/common/core/types.hpp"
#include "core/pyramid.hpp"
#include "core/rect.hpp"
#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"
#include "libvideostitch/status.hpp"
#include "libvideostitch/ptv.hpp"

#include <vector>
#include <map>
#include <memory>

namespace VideoStitch {
namespace Core {

#define MAX_INVALID_COST 100000

}  // namespace Core
}  // namespace VideoStitch
