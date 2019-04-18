// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/processors/maskoverlay.hpp"

namespace VideoStitch {
namespace Core {

Status maskOverlay(GPU::Surface& /*dst*/, unsigned /*width*/, unsigned /*height*/, uint32_t /*color*/,
                   GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Mask overlay not implemented in OpenCL backend"};
}

}  // namespace Core
}  // namespace VideoStitch
