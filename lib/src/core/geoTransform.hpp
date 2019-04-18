// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backend/cpp/core/transformTypes.hpp"

#include "coordinates.hpp"
#include "transformGeoParams.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Core {

class InputDefinition;

namespace TransformStack {

/**
 * @brief Class that geometric transformation on the CPU.
 */
class GeoTransform {
 public:
  /**
   * Creates a Transform that maps the given input into the given panorama.
   * @param pano The PanoDefinition
   * @param im The InputDefinition
   */
  static GeoTransform* create(const PanoDefinition& pano, const InputDefinition& im);

  virtual ~GeoTransform();

  /**
   * Transforms a single point on the CPU from panorama to rig unit sphere.
   * @param uv Coordinates, relative to the center of the panorama.
   * @returns Coordinates in rig sphere.
   */
  SphericalCoords3 mapPanoramaToRigSpherical(CenterCoords2 uv) const;

  /**
   * Transforms a single point on the CPU from panorama to input coordinates.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param uv Coordinates, relative to the center of the panorama.
   * @param time time of the geometry
   * @returns Coordinates in input space, relative to the center of the input.
   */
  CenterCoords2 mapPanoramaToInput(const InputDefinition& im, CenterCoords2 uv, int time) const;

  /**
   * Transforms a single point on the CPU from input to panorama coordinates.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param uv coordinates in input space, relative to the center of the input.
   * @param time time of the geometry
   * @returns Coordinates in spherical space.
   */
  CenterCoords2 mapInputToPanorama(const InputDefinition& im, CenterCoords2 uv, int time) const;

  /**
   * Transforms a single point on the CPU from spherical rig-center based coordinates to input coordinates.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param pt coordinates in spherical rig space.
   * @param time time of the geometry
   * @returns Coordinates in input space, relative to the center of the input.
   */
  CenterCoords2 mapRigSphericalToInput(const InputDefinition& im, const SphericalCoords3& pt, int time) const;

  /**
   * Transforms a single point on the CPU from input to spherical rig-center based coordinates.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param uv coordinates in input space, relative to the center of the input.
   * @param time time of the geometry
   * @param radius radius of the rig sphere (note that this is not the camera sphere if the camera has a translation off
   * the rig center)
   * @returns Coordinates in spherical space.
   * @note radius should be large enough to contain the camera center of projection
   * @note the minimum radius value is given by getInputMinimumRigSphereRadius()
   */
  SphericalCoords3 mapInputToRigSpherical(const InputDefinition& im, CenterCoords2 uv, int time,
                                          float rigSphereRadius) const;

  /**
   *  Same transformation, using sphereScale from PanoDef
   */
  SphericalCoords3 mapInputToRigSpherical(const InputDefinition& im, CenterCoords2 uv, int time) const {
    return mapInputToRigSpherical(im, uv, time, rigSphereRadius);
  }

  /**
   * Transforms a single point on the CPU from input to its camera sphere. Coordinates translated to rig-center base.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param uv coordinates in input space, relative to the center of the input.
   * @param time time of the geometry
   * @param radius radius of the sphere
   * @returns Coordinates in spherical rig space.
   */
  SphericalCoords3 mapInputToScaledCameraSphereInRigBase(const InputDefinition& im, CenterCoords2 uv, int time,
                                                         float cameraSphereRadius = 1.0f) const;

  /**
   * Gets the minimum rig sphere radius that contains the center of projection of the camera
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param time time of the geometry
   * @note calling mapInputToRigSpherical() with a radius value smaller than this one is an error, an assert will fail
   * in debug mode
   */
  float computeInputMinimumRigSphereRadius(const InputDefinition& im, int time) const;

  /**
   * Returns true if a point is within bounds.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param uv coordinates in input space, relative to the top-left of the input.
   */
  bool isWithinInputBounds(const InputDefinition& im, TopLeftCoords2 uv) const;

 protected:
  GeoTransform(const GeoTransform& other);
  GeoTransform();

 private:
  float2 distort(float2 uv, const float2 inputScale, const vsDistortion distortion, const float2 centerShift) const;

  float3 mapInputToCameraSpherical(const InputDefinition& im, const GeometryDefinition& geometry,
                                   const TransformGeoParams& params, const CenterCoords2 uv) const;

  // mapping implementation shared with GPU
  float2 mapPanoramaToInput(float2 uv, const float2 panoScale, const vsfloat3x4 pose, const float2 inputScale,
                            const vsDistortion distortion, const float2 centerShift) const;

  float2 mapInputToPanorama(float2 uv, const float2 panoScale, const vsfloat3x4 poseInverse,
                            const float rigSphereRadius, const float2 inputScale, const vsDistortion distortion,
                            const float2 centerShift) const;

  float2 mapRigSphericalToInput(float3 pt, const vsfloat3x4 pose, const float2 inputScale,
                                const vsDistortion distortion, const float2 centerShift) const;

  float3 mapInputToCameraSphere(float2 uv, const float2 inputScale, const vsDistortion distortion,
                                const float2 centerShift) const;

  float3 mapInputToRigSpherical(float2 uv, const vsfloat3x4 poseInverse, const float rigSphereRadius,
                                const float2 inputScale, const vsDistortion distortion, const float2 centerShift) const;

  float3 tracePointToRigSphere(float3 pt, const vsfloat3x4 poseInverse, const float rigSphereRadius) const;

  // Host-side transform functions
  Convert2D3DFnT fromOutputToSphereHostFn;
  Convert3D2DFnT fromSphereToInputHostFn;
  Convert3D2DFnT fromSphereToOutputHostFn;
  Convert2D3DFnT fromInputToSphereHostFn;
  IsWithinFnT isWithinHostFn;
  DistortionTransformFnt distortionTransformMetersHostFn;
  DistortionTransformFnt distortionTransformPixelsHostFn;
  DistortionTransformFnt inverseDistortionTransformMetersHostFn;
  DistortionTransformFnt inverseDistortionTransformPixelsHostFn;

  float2 panoScale;
  float rigSphereRadius;
};

}  // namespace TransformStack
}  // namespace Core
}  // namespace VideoStitch
