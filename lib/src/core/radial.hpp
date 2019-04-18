// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <cmath>

#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/projections.hpp"

namespace VideoStitch {
namespace Core {
void computeRadialParams(const InputDefinition& im, const GeometryDefinition& geometry, float& radial0, float& radial1,
                         float& radial2, float& radial3, float& radial4);

/**
 * Compute the 5th radial parameter.
 */
double computeRadial4(double radial0, double radial1, double radial2, double radial3);

}  // namespace Core
}  // namespace VideoStitch
