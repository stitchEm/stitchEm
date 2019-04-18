// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/depthDef.hpp"

#include <algorithm>

namespace VideoStitch {
namespace Core {

DepthDefinition::DepthDefinition() : numPyramidLevels(1) {}

DepthDefinition::~DepthDefinition() {}

int DepthDefinition::getNumPyramidLevels() const { return numPyramidLevels; }

void DepthDefinition::setNumPyramidLevels(int numLevels) {
  // minimum numLevels is 1
  numPyramidLevels = std::max(numLevels, 1);
}

bool DepthDefinition::isMultiScale() const { return numPyramidLevels >= 2; }

}  // namespace Core
}  // namespace VideoStitch
