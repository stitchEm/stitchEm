// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/coredepth/sweep.hpp"

#include "backend/common/coredepth/sphereSweepParams.h"

#include "../context.hpp"
#include "../kernel.hpp"
#include "../surface.hpp"

#include "core/transformGeoParams.hpp"
#include "core/surfacePyramid.hpp"

#include "libvideostitch/geometryDef.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/depthDef.hpp"

namespace VideoStitch {
namespace GPU {

namespace {
// #include "sphereSweep.xxd"
}

// INDIRECT_REGISTER_OPENCL_PROGRAM(sphereSweep, true);

Status splatInputWithDepthIntoPano(const Core::PanoDefinition& /*panoDef*/, Core::PanoSurface& /*pano*/,
                                   const GPU::Surface& /*depthSurface*/,
                                   const std::map<videoreaderid_t, Core::SourceSurface*>& /*surfaces*/,
                                   GPU::Stream /*stream*/) {
  // TODO FIXME LATER
  return {Origin::GPU, ErrType::ImplementationError, "Depth splatting not implemented for OpenCL yet"};
}

Status sphereSweepInput(videoreaderid_t /*sourceID*/, int /*frame*/, GPU::Surface& /*dst*/,
                        const std::map<videoreaderid_t, Core::SourceSurface*>& /*surfaces*/,
                        const Core::PanoDefinition& /*panoDef*/, GPU::Stream& /*stream*/, const float /*scale*/) {
  // TODO FIXME LATER
  return {Origin::GPU, ErrType::ImplementationError, "Sphere sweep not implemented for OpenCL yet"};
}

// Implementation for command line tool
// Sweeps into input-sized surface and regularizes through SGM
Status sphereSweepInputSGM(videoreaderid_t /*sourceID*/, int /*frame*/, GPU::Surface& /*dst*/,
                           GPU::HostBuffer<unsigned short>& /*hostCostVolume*/,
                           const std::map<videoreaderid_t, Core::SourceSurface*>& /*surfaces*/,
                           const Core::PanoDefinition& /*panoDef*/, GPU::Stream& /*stream*/, const float /*scale*/) {
  // TODO FIXME LATER
  return {Origin::GPU, ErrType::ImplementationError, "SGM not implemented for OpenCL yet"};
}

// Implementation for command line tool
// Converts SGM disparity to depth values in dst
Status sphereSweepInputDisparityToDepth(videoreaderid_t /*sourceID*/, int /*frame*/, GPU::Surface& /*dst*/,
                                        GPU::HostBuffer<short>& /*hostDisparity*/, bool /*useHostDisparity*/,
                                        const std::map<videoreaderid_t, Core::SourceSurface*>& /*surfaces*/,
                                        const Core::PanoDefinition& /*panoDef*/, GPU::Stream& /*stream*/,
                                        const float /*scale*/) {
  // TODO FIXME LATER
  return {Origin::GPU, ErrType::ImplementationError, "SGM not implemented for OpenCL yet"};
}

Status sphereSweepInputStep(videoreaderid_t /*sourceID*/, int /*frame*/, GPU::Surface& /*dstDepth*/,
                            GPU::Surface& /*lowerLevelDepth*/,
                            const std::map<videoreaderid_t, Core::SourceSurface*>& /*surfaces*/,
                            const Core::PanoDefinition& /*panoDef*/, GPU::Stream& /*stream*/, const float /*scale*/) {
  // TODO FIXME LATER
  return {Origin::GPU, ErrType::ImplementationError, "Sphere sweep not implemented for OpenCL yet"};
}

Status sphereSweepInputSGMSingleScale(videoreaderid_t /*sourceID*/, int /*frame*/, GPU::Surface& /*dst*/,
                                      const std::map<videoreaderid_t, Core::SourceSurface*>& /*inputSurfaces*/,
                                      const Core::InputDefinition& /*inputDef*/,
                                      const Core::PanoDefinition& /*panoDef*/, GPU::Stream& /*stream*/,
                                      const float /*scale*/, const int /*uniquenessRatio*/) {
  // TODO FIXME LATER
  return {Origin::GPU, ErrType::ImplementationError, "Sphere sweep not implemented for OpenCL yet"};
}

Status sphereSweepInputSGMMultiScale(videoreaderid_t /*sourceID*/, int /*frame*/, GPU::Surface& /*dst*/,
                                     const std::vector<Core::InputPyramid>& /*inputPyramids*/,
                                     const Core::DepthPyramid& /*depthPyramid*/,
                                     const Core::InputDefinition& /*inputDef*/, const Core::PanoDefinition& /*panoDef*/,
                                     const Core::DepthDefinition& /*depthDef*/, GPU::Stream& /*stream*/) {
  // TODO FIXME LATER
  return {Origin::GPU, ErrType::ImplementationError, "Sphere sweep not implemented for OpenCL yet"};
}

int maxDepthInputs() { return NUM_INPUTS; }

int numSphereSweeps() { return numSphereScales; }

}  // namespace GPU
}  // namespace VideoStitch
