// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "core/surfacePyramid.hpp"

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

namespace VideoStitch {

namespace GPU {
class Surface;
}

namespace Core {

class DepthDefinition;
class InputDefinition;
class PanoDefinition;
class PanoSurface;
class SourceSurface;

enum class SGMPostProcessing { Off, On };

enum class BilateralFilterPostProcessing { Off, On };

// Implementation from prototype:
// sphere sweep into pano output
// can be previewed in Studio, when using sphere_sweep merger
// does depth estimation for the first input (reference), then reprojects the input with the depth
// rendering camera is moving a bit with each call
Status sphereSweepIntoPano(const PanoDefinition& panoDef, const Core::DepthDefinition& depthDef, PanoSurface& pano,
                           const std::map<videoreaderid_t, Core::SourceSurface*>& inputSurfaces,
                           std::vector<InputPyramid>& pyramids, const DepthPyramid& depthPyramid, GPU::Stream stream);

// Implementation for command line tool
// Sweeps into input-sized surface
Status sphereSweepInputMultiScale(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                                  const std::vector<InputPyramid>& inputPyramids, const DepthPyramid& depthPyramid,
                                  const PanoDefinition& panoDef, const DepthDefinition& depthDef,
                                  const SGMPostProcessing sgmPostProcessing,
                                  const BilateralFilterPostProcessing bilateralFilterPostProcessing,
                                  GPU::Stream& stream);

}  // namespace Core
}  // namespace VideoStitch
