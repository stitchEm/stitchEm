// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backend/cpp/core/transformTypes.hpp"

#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/projections.hpp"

namespace VideoStitch {
namespace Core {

class PanoDefinition;

/**
 * The geometric params for a transform.
 */
class TransformGeoParams {
 public:
  /**
   * Constructor
   * @param im The InputDefinition
   * @param geometry the geometryDefinition to use
   */
  TransformGeoParams(const InputDefinition& im, const GeometryDefinition& geometry, const double sphereScale);
  /**
   * Constructor
   * @param im The InputDefinition
   * @param geometry the geometryDefinition to use
   * @param pano the PanoDefinition to use
   */
  TransformGeoParams(const InputDefinition& im, const GeometryDefinition& geometry, const PanoDefinition& pano);

  /**
   * Computes the sphereDist for a pano.
   * @param proj output projection
   * @param panoWidth pano width
   * @param panoHFOVDeg Horizontal field of view
   * @returns sphere dist.
   */
  static float computePanoScale(const PanoProjection& proj, const int64_t panoWidth, const float panoHFOVDeg);

  static double getInverseDemiDiagonalSquared(const InputDefinition& im);

  /**
   * Computes the input horizontal scale parameter.
   * @param im input
   * @param fov the fov to use
   */
  static float computeHorizontalScale(const InputDefinition& im, double fov);

  /**
   * Computes the fov from the focal value
   * @param im input
   * @param focal the focal to use
   */
  static double computeFov(const InputDefinition& im, double focal);

  /**
   * Computes the input vertical scale parameter.
   * @param im input
   * @param fov the fov to use
   */
  static float computeVerticalScale(const InputDefinition& im, double fov);

  /**
   * Computes the input scale parameter.
   */
  static float computeInputScale(InputDefinition::Format imFmt, int64_t imWidth, float imHFOVDeg);

  /**
   * Computes the fov from the focal value
   */
  static double computeFovFromFocal(InputDefinition::Format imFmt, int64_t imWidth, double focal);

  /**
  Get the pose matrix elements (row major)
  */
  const vsfloat3x4& getPose() const { return pose; }

  const vsfloat3x4 getPoseScaled(const float scale) const {
    vsfloat3x4 poseScaled;
    /* rescale rotation values only */
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        poseScaled.values[row][col] = pose.values[row][col] * scale;
      }
      poseScaled.values[row][3] = pose.values[row][3];
    }
    return poseScaled;
  }

  /**
  Get the pose matrix inversed elements (row major)
  */
  const vsfloat3x4& getPoseInverse() const { return poseInverse; }

  const vsfloat3x4 getPoseInverseScaled(const float scale) const {
    vsfloat3x4 poseScaled;
    /* rescale rotation values only */
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        poseScaled.values[row][col] = poseInverse.values[row][col] * scale;
      }
      poseScaled.values[row][3] = poseInverse.values[row][3];
    }
    return poseScaled;
  }

  /**
  Get the Radial and Non-Radial distortion elements
  */
  const vsDistortion& getDistortion() const { return distortion; }

 private:
  /**
  3D pose matrix
  */
  vsfloat3x4 pose;

  /**
  3D pose matrix
  */
  vsfloat3x4 poseInverse;

  /**
  Coefficients for the distortion
  */
  vsDistortion distortion;
};

}  // namespace Core
}  // namespace VideoStitch
