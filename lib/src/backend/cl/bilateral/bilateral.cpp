// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "bilateral/bilateral.hpp"

namespace VideoStitch {
namespace GPU {

// #include "backend/common/bilateral/bilateral.gpu"

Status depthJointBilateralFilter(GPU::Surface& /*output*/, const GPU::Surface& /*input*/,
                                 const Core::SourceSurface& /*textureSurface*/, GPU::Stream& /*stream*/) {
  // TODO FIXME LATER
  return {Origin::GPU, ErrType::ImplementationError, "Depth bilateral filter not implemented for OpenCL yet"};
}

}  // namespace GPU
}  // namespace VideoStitch
