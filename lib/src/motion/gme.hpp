// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/config.hpp"
#include "libvideostitch/quaternion.hpp"

#include "gpu/vectorTypes.hpp"

#include <map>
#include <vector>

namespace VideoStitch {
namespace Motion {
namespace ImageSpace {

struct MotionVector {
  MotionVector() {
    from.x = 0;
    from.y = 0;
    to.x = 0;
    to.y = 0;
  }

  MotionVector(float2 from, float2 to) : from(from), to(to) {}

  inline float magnitude2() const {
    float diffX = to.x - from.x;
    float diffY = to.y - from.y;
    return (diffX * diffX) + (diffY * diffY);
  }

  float2 from;
  float2 to;
};

typedef std::vector<MotionVector> MotionVectorField;
typedef std::map<int64_t, MotionVectorField> MotionVectorFieldTimeSeries;
}  // namespace ImageSpace

namespace SphericalSpace {

struct MotionVector {
  MotionVector(const Quaternion<double>& from, const Quaternion<double>& to) : from(from), to(to) {}

  Quaternion<double> from;
  Quaternion<double> to;
};

typedef std::vector<MotionVector> MotionVectorField;
typedef std::map<int64_t, MotionVectorField> MotionVectorFieldTimeSeries;
}  // namespace SphericalSpace
}  // namespace Motion
}  // namespace VideoStitch
