// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PROJECTIONS_HPP_
#define PROJECTIONS_HPP_

#include "config.hpp"

#include <string>

namespace VideoStitch {
namespace Core {

/**
 * nvcc on mac does not currently support c++11 enum class. This looks like it as far as syntax is concerned,
 * so we'll be able to just replace all this by an enum class when possible.
 * Note that this is not type-safe though.
 */
class VS_EXPORT PanoProjection {
 public:
  /**
   * A pano projection.
   * WARNING Never change these values.
   */
  enum Type {
    Rectilinear = 0,
    Cylindrical = 1,
    Equirectangular = 2,
    FullFrameFisheye = 3,
    Stereographic = 4,
    CircularFisheye = 5,
    Cubemap = 6,
    EquiangularCubemap = 7
  };

  /**
   * Default construction uses a random value.
   */
  PanoProjection() : value(Equirectangular) {}

  /**
   * A PanoProjection is implicitly constructible from the enum.
   * @param value enum value
   */
  PanoProjection(Type value) : value(value) {}

  /**
   * A PanoProjection is implicitly convertible to the enum.
   */
  operator Type() const { return value; }

 private:
  Type value;
};

/**
 * Returns the PTV name of an output projection.
 * @param proj projection
 */
VS_EXPORT const char* getPanoProjectionName(const PanoProjection& proj);

/**
 * Returns the projection with this PTV name.
 * @param name The format name, e.g. "equirectangular".
 * @param proj The projection
 * @returns false if this is not a valid name.
 */
bool VS_EXPORT getPanoProjectionFromName(const std::string& name, PanoProjection& proj);

}  // namespace Core
}  // namespace VideoStitch

#endif
