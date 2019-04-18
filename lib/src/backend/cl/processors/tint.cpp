// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/processors/tint.hpp"

namespace VideoStitch {
namespace Core {

Status tint(GPU::Surface& /*dst*/, unsigned /*width*/, unsigned /*height*/, int32_t /*r*/, int32_t /*g*/, int32_t /*b*/,
            GPU::Stream stream) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Tinting not implemented in OpenCL backend"};
}

}  // namespace Core
}  // namespace VideoStitch
