// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "panoDimensionsDef.hpp"

#include "transformGeoParams.hpp"

#include "libvideostitch/panoDef.hpp"

namespace VideoStitch {
namespace Core {

inline PanoDimensions getPanoDimensions(const PanoDefinition& pano) {
  PanoDimensions p;
  p.width = (int32_t)pano.getWidth();
  p.height = (int32_t)pano.getHeight();
  p.scaleX = TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getWidth(), 360.f);
  p.scaleY = 2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getHeight(), 360.f);
  return p;
};

}  // namespace Core
}  // namespace VideoStitch
