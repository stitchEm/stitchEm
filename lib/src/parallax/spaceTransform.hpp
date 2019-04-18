// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backend/common/core/types.hpp"
#include "backend/cpp/core/transformTypes.hpp"

#include "core/coordinates.hpp"
#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/matrix.hpp"

#include <stdint.h>
#include <vector>

namespace VideoStitch {
namespace Core {

class InputDefinition;
class PanoDefinition;
class GeoTransform;
struct Rect;

/**
 * This class is used to pre-compute coordinate mappings from an input to an output space and vice versa.
 * It deals with 2 different coordinate spaces:
 *  1) Input space: the space used for capturing the input videos.
 *  2) Output space: the intermediate space, constructed by rotating the original pole @oldPole of the original
 * panoramic space to a new pole @newPole.
 */

class SpaceTransform {
 public:
  /**
   * Creates a Transform that maps the given input into the given panorama.
   * @param pano The PanoDefinition
   * @param im The InputDefinition
   */
  static SpaceTransform* create(const InputDefinition& im, const Vector3<double> oldPole,
                                const Vector3<double> newPole);

  /**
   * Find the average Cartesian coordinate of an input.
   * @param pano The PanoDefinition
   * @param im The InputDefinition
   * @param sampleCount Number of samples used to calculate the average coordinate
   */
  static Vector3<double> getAverageSphericalCoord(const PanoDefinition& pano, const InputDefinition& im,
                                                  const int sampleCount = 100);
  static Vector3<double> normalizedSphereToCartesian(const SphericalCoords2& sphericalCoord);

  const vsfloat3x4& getPose() const { return pose; }

  const vsfloat3x4& getPoseInverse() const { return poseInverse; }

  const vsfloat3x4 getCombinedPose(const vsfloat3x4& input) const {
    vsfloat3x4 combinedPose;
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        float v = 0.0f;
        for (int t = 0; t < 3; t++) {
          v += pose.values[row][t] * input.values[t][col];
        }
        combinedPose.values[row][col] = v;
      }
      combinedPose.values[row][3] = input.values[row][3];
    }
    return combinedPose;
  }

  const vsfloat3x4 getCombinedInversePose(const vsfloat3x4& input) const {
    vsfloat3x4 combinedPoseInverse;
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        float v = 0.0f;
        for (int t = 0; t < 3; t++) {
          v += input.values[row][t] * poseInverse.values[t][col];
        }
        combinedPoseInverse.values[row][col] = v;
      }
      combinedPoseInverse.values[row][3] = input.values[row][3];
    }
    return combinedPoseInverse;
  }

  /**
   * Find the average spherical coordinate of an input.
   * @param pano The PanoDefinition
   * @param im The InputDefinition
   * @param oldOriCoord The old origin-coord in Cartesian coordinate (as define in normalizedSphereToNormalizedErect )
   * @param newOriCoord The new origin-coord in Cartesian coordinate. A rotation matrix is generated from this to rotate
   * from old --> new origin
   */
  SpaceTransform(const Vector3<double>& oldOriCoord, const Vector3<double>& newOriCoord);

  /**
   * Finds pixel coordinate of inputs in the final setup
   */
  virtual Status mapCoordInputToOutput(const int time, GPU::Buffer<float2> outputBuffer, const int inputWidth,
                                       const int inputHeight, const GPU::Buffer<const float2> inputBuffer,
                                       const GPU::Buffer<const uint32_t> inputMask, const PanoDefinition& pano,
                                       const videoreaderid_t id, GPU::Stream gpuStream) const = 0;

  /**
   * Finds pixel coordinate of inputs in the final setup
   */
  virtual Status mapCoordOutputToInput(const int time, const int offsetX, const int offsetY, const int croppedWidth,
                                       const int croppedHeight, GPU::Buffer<float2> outputBuffer,
                                       GPU::Buffer<uint32_t> maskBuffer, const PanoDefinition& pano,
                                       const videoreaderid_t id, GPU::Stream gpuStream) const = 0;

 protected:
  vsfloat3x4 pose;
  vsfloat3x4 poseInverse;
};
}  // namespace Core
}  // namespace VideoStitch
