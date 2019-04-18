// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/input/maskInput.hpp"

namespace VideoStitch {
namespace Input {

Status maskInput(GPU::Surface& /*dst*/, GPU::Buffer<const unsigned char> /*maskDevBufferP*/, unsigned /*width*/,
                 unsigned /*height*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Masking input not implemented in OpenCL backend"};
}

}  // namespace Input
}  // namespace VideoStitch
