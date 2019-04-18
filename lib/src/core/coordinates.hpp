// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef COORDINATES_HPP_
#define COORDINATES_HPP_

#include "gpu/vectorTypes.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Core {

class CenterCoords2;

/**
 * Coordinates from the top-left of an image.
 */
class TopLeftCoords2 {
 public:
  TopLeftCoords2() : x(0.f), y(0.f) {}
  TopLeftCoords2(float x, float y) : x(x), y(y) {}

  TopLeftCoords2(CenterCoords2 topLeft, TopLeftCoords2 center);

  /**
   * Use with care.
   */
  float2 toFloat2() const { return make_float2(x, y); }

  float x;
  float y;
};

/**
 * Coordinates from the center of an image.
 */
class CenterCoords2 {
 public:
  CenterCoords2(float x, float y) : x(x), y(y) {}
  explicit CenterCoords2(float2 uv) : x(uv.x), y(uv.y) {}

  CenterCoords2(TopLeftCoords2 topLeft, TopLeftCoords2 center) : x(topLeft.x - center.x), y(topLeft.y - center.y) {}

  /**
   * Use with care.
   */
  float2 toFloat2() const { return make_float2(x, y); }

  float x;
  float y;
};

inline TopLeftCoords2::TopLeftCoords2(CenterCoords2 topLeft, TopLeftCoords2 center)
    : x(topLeft.x + center.x), y(topLeft.y + center.y) {}

/**
 * Spherical coordinates.
 */
class SphericalCoords2 {
 public:
  SphericalCoords2(float x, float y) : x(x), y(y) {}
  explicit SphericalCoords2(float2 uv) : x(uv.x), y(uv.y) {}

  /**
   * Use with care.
   */
  float2 toFloat2() const { return make_float2(x, y); }

  // Operators
  SphericalCoords2& operator+=(const SphericalCoords2& rhs) {
    this->x += rhs.x;
    this->y += rhs.y;
    return *this;
  }

  SphericalCoords2& operator/=(const float rhs) {
    this->x /= rhs;
    this->y /= rhs;
    return *this;
  }

  float x;
  float y;
};

/**
 * Spherical coordinates.
 */
class SphericalCoords3 {
 public:
  SphericalCoords3(float x, float y, float z) : x(x), y(y), z(z) {}
  explicit SphericalCoords3(float3 pt) : x(pt.x), y(pt.y), z(pt.z) {}

  /**
   * Use with care.
   */
  float3 toFloat3() const { return make_float3(x, y, z); }

  // Operators
  SphericalCoords3& operator+=(const SphericalCoords3& rhs) {
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    return *this;
  }

  SphericalCoords3& operator*=(const float rhs) {
    this->x *= rhs;
    this->y *= rhs;
    this->z *= rhs;
    return *this;
  }

  SphericalCoords3& operator/=(const float rhs) {
    this->x /= rhs;
    this->y /= rhs;
    this->z /= rhs;
    return *this;
  }

  float x;
  float y;
  float z;
};

}  // namespace Core
}  // namespace VideoStitch

#endif
