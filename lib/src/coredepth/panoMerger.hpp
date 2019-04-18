// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/stream.hpp"

#include "libvideostitch/status.hpp"

#include <vector>

namespace VideoStitch {

namespace Core {

class PanoDefinition;
class PanoSurface;

class SourceSurface;

class PanoMerger {
 public:
  /**
   * Creates a pano merger
   */
  PanoMerger();

  virtual ~PanoMerger();

  /**
   TODO
   */
  virtual Status computeAsync(const PanoDefinition& panoDef, PanoSurface& pano,
                              const std::map<videoreaderid_t, Core::SourceSurface*>& surfaces, GPU::Stream stream) = 0;
};

}  // namespace Core
}  // namespace VideoStitch
