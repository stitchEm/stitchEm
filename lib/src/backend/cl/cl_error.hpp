// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/status.hpp"
#include "libvideostitch/logging.hpp"
#include "opencl.h"

#define CL_ERROR(stmt) VideoStitch::GPU::checkErrorStatus((stmt), __FILE__, __LINE__)
#define PROPAGATE_CL_ERR(stmt) \
  { PROPAGATE_FAILURE_STATUS(CL_ERROR(stmt)); }

namespace VideoStitch {
namespace GPU {

Status checkErrorStatus(int errorCode, const char *file, int line);

}
}  // namespace VideoStitch
