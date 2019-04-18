// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/hostBuffer.hpp"
#include "gpu/stream.hpp"

namespace VideoStitch {

namespace Core {

class InputDefinition;
class PanoDefinition;
class PanoSurface;
class SourceSurface;
}  // namespace Core

namespace GPU {

class Surface;

Status splatInputWithDepthIntoPano(const Core::PanoDefinition& panoDef, Core::PanoSurface& pano,
                                   const GPU::Surface& depthSurface,
                                   const std::map<videoreaderid_t, Core::SourceSurface*>& surfaces, GPU::Stream stream);

Status sphereSweepInput(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                        const std::map<videoreaderid_t, Core::SourceSurface*>& surfaces,
                        const Core::PanoDefinition& panoDef, GPU::Stream& stream, const float scale = 1.0f);

// Implementation for command line tool
// Sweeps into input-sized surface and regularizes through SGM
Status sphereSweepInputSGM(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                           GPU::HostBuffer<unsigned short>& hostCostVolume,
                           const std::map<videoreaderid_t, Core::SourceSurface*>& surfaces,
                           const Core::PanoDefinition& panoDef, GPU::Stream& stream, const float scale = 1.0f);

// Implementation for command line tool
// Converts SGM disparity to depth values in dst
Status sphereSweepInputDisparityToDepth(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                                        GPU::HostBuffer<short>& hostDisparity, bool useHostDisparity,
                                        const std::map<videoreaderid_t, Core::SourceSurface*>& surfaces,
                                        const Core::PanoDefinition& panoDef, GPU::Stream& stream,
                                        const float scale = 1.0f);

Status sphereSweepInputStep(videoreaderid_t sourceID, int frame, GPU::Surface& dstDepth, GPU::Surface& lowerLevelDepth,
                            const std::map<videoreaderid_t, Core::SourceSurface*>& surfaces,
                            const Core::PanoDefinition& panoDef, GPU::Stream& stream, const float scale = 1.0f);

// returns the number of sphereSweeps() (for cost allocation reasons)
int numSphereSweeps();

// returns the maximum number of inputs for depth estimation
int maxDepthInputs();

}  // namespace GPU
}  // namespace VideoStitch
