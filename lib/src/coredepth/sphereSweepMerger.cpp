// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "sphereSweepMerger.hpp"

#include "coredepth/sphereSweep.hpp"

#include "libvideostitch/depthDef.hpp"

static const int numLevels = 4;

namespace VideoStitch {
namespace Core {

// TODO depth pyramid hardcoded to size of first video input
SphereSweepMerger::SphereSweepMerger(const PanoDefinition& panoDef)
    : depthPyramid{(int)numLevels, (int)panoDef.getVideoInput(0).getWidth(),
                   (int)panoDef.getVideoInput(0).getHeight()} {
  for (const InputDefinition& videoInputDef : panoDef.getVideoInputs()) {
    InputPyramid pyramid{(int)numLevels, (int)videoInputDef.getWidth(), (int)videoInputDef.getHeight()};
    pyramids.push_back(std::move(pyramid));
  }

  // TODO propagate out of memory fail
}

Status SphereSweepMerger::computeAsync(const PanoDefinition& panoDef, PanoSurface& pano,
                                       const std::map<videoreaderid_t, Core::SourceSurface*>& surfaces,
                                       GPU::Stream stream) {
  for (videoreaderid_t i = 0; i < panoDef.numVideoInputs(); i++) {
    pyramids[i].compute(surfaces.at(i), stream);
  }

  // TODO as merger parameter
  DepthDefinition depthDef{};
  depthDef.setNumPyramidLevels(numLevels);

  return sphereSweepIntoPano(panoDef, depthDef, pano, surfaces, pyramids, depthPyramid, stream);
}

}  // namespace Core
}  // namespace VideoStitch
