// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/projections.hpp"

namespace VideoStitch {
namespace Core {

const char* getPanoProjectionName(const PanoProjection& proj) {
  PanoProjection::Type ptype = proj;
  switch (ptype) {
    case PanoProjection::Rectilinear:
      return "rectilinear";
    case PanoProjection::Cylindrical:
      return "cylindrical";
    case PanoProjection::Equirectangular:
      return "equirectangular";
    case PanoProjection::FullFrameFisheye:
      return "ff_fisheye";
    case PanoProjection::Stereographic:
      return "stereographic";
    case PanoProjection::CircularFisheye:
      return "circular_fisheye";
    case PanoProjection::Cubemap:
      return "cubemap";
    case PanoProjection::EquiangularCubemap:
      return "equiangular_cubemap";
  }
  return NULL;
}

bool getPanoProjectionFromName(const std::string& name, PanoProjection& proj) {
  if (!name.compare("rectilinear")) {
    proj = PanoProjection(PanoProjection::Rectilinear);
    return true;
  } else if (!name.compare("equirectangular")) {
    proj = PanoProjection(PanoProjection::Equirectangular);
    return true;
  } else if (!name.compare("ff_fisheye")) {
    proj = PanoProjection(PanoProjection::FullFrameFisheye);
    return true;
  } else if (!name.compare("stereographic")) {
    proj = PanoProjection(PanoProjection::Stereographic);
    return true;
  } else if (!name.compare("circular_fisheye")) {
    proj = PanoProjection(PanoProjection::CircularFisheye);
    return true;
  } else if (!name.compare("cubemap")) {
    proj = PanoProjection(PanoProjection::Cubemap);
    return true;
  } else if (!name.compare("equiangular_cubemap")) {
    proj = PanoProjection(PanoProjection::EquiangularCubemap);
    return true;
  }
  return false;
}

}  // namespace Core
}  // namespace VideoStitch
