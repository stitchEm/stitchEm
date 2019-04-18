// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/rigDef.hpp"

#include "rigGraph.hpp"

namespace VideoStitch {
namespace Calibration {

class Camera;

class RigBuilder {
 public:
  static double build(std::vector<std::shared_ptr<Camera> >& cameras, const RigGraph& rigGraph,
                      const unsigned int center);
};

}  // namespace Calibration
}  // namespace VideoStitch
