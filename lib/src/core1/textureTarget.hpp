// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <string>

namespace VideoStitch {
namespace Core {

enum TextureTarget {
  EQUIRECTANGULAR = 0,
  CUBE_MAP_POSITIVE_X = 1,
  CUBE_MAP_NEGATIVE_X = 2,
  CUBE_MAP_POSITIVE_Y = 3,
  CUBE_MAP_NEGATIVE_Y = 4,
  CUBE_MAP_POSITIVE_Z = 5,
  CUBE_MAP_NEGATIVE_Z = 6,
  EQUIRECTANGULAR_LOOKUP = 7,
  TEXTURE_TARGET_SIZE = 8
};

inline std::string toString(TextureTarget t) {
  switch (t) {
    case EQUIRECTANGULAR:
      return "equirectangular";
    case CUBE_MAP_POSITIVE_X:
      return "face_+x";
    case CUBE_MAP_NEGATIVE_X:
      return "face_-x";
    case CUBE_MAP_POSITIVE_Y:
      return "face_+y";
    case CUBE_MAP_NEGATIVE_Y:
      return "face_-y";
    case CUBE_MAP_POSITIVE_Z:
      return "face_+z";
    case CUBE_MAP_NEGATIVE_Z:
      return "face_-z";
    default:
      return "error";
  }
}

}  // namespace Core
}  // namespace VideoStitch
