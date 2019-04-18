// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "geoTransform.hpp"

#include "backend/cpp/core/transformStack.hpp"

#include "radial.hpp"
#include "common/angles.hpp"

#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/geometryDef.hpp"

#define HOST_TRANSFORM
#include "kernels/withinStack.cu"
#undef HOST_TRANSFORM

#include <memory>

#define RETURN_3D_IF_INVALID_INVERSE_2D(uv)                                                      \
  if (uv.x == INVALID_INVERSE_DISTORTION && uv.y == INVALID_INVERSE_DISTORTION) {                \
    return {INVALID_INVERSE_DISTORTION, INVALID_INVERSE_DISTORTION, INVALID_INVERSE_DISTORTION}; \
  }

#define RETURN_2D_IF_INVALID_INVERSE_3D(pt)                                       \
  if (pt.x == INVALID_INVERSE_DISTORTION && pt.y == INVALID_INVERSE_DISTORTION && \
      pt.z == INVALID_INVERSE_DISTORTION) {                                       \
    return {INVALID_INVERSE_DISTORTION, INVALID_INVERSE_DISTORTION};              \
  }

#define RETURN_3D_IF_INVALID_INVERSE_3D(pt)                                                      \
  if (pt.x == INVALID_INVERSE_DISTORTION && pt.y == INVALID_INVERSE_DISTORTION &&                \
      pt.z == INVALID_INVERSE_DISTORTION) {                                                      \
    return {INVALID_INVERSE_DISTORTION, INVALID_INVERSE_DISTORTION, INVALID_INVERSE_DISTORTION}; \
  }

namespace VideoStitch {
namespace Core {

namespace TransformStack {

#define FUNCTION_NAME_3(a, b, c) GeoTransform::a
#define FUNCTION_NAME_4(a, b, c, d) GeoTransform::a

#define const_member const

#define sqrtf_vs std::sqrt

#define fromOutputToSphere fromOutputToSphereHostFn
#define fromSphereToInput fromSphereToInputHostFn
#define distortionMetersTransform distortionTransformMetersHostFn
#define distortionPixelsTransform distortionTransformPixelsHostFn

#include "backend/common/core1/distort.gpu.incl"
#include "backend/common/core1/mapFunction.gpu.incl"

#define fromInputToSphere fromInputToSphereHostFn
#define fromSphereToOutput fromSphereToOutputHostFn
#define inverseRadialPixelsTransform inverseDistortionTransformPixelsHostFn
#define inverseDistortionMetersTransform inverseDistortionTransformMetersHostFn
#include "backend/common/parallax/mapInverseFunction.gpu.incl"

#undef const_member

#undef sqrtf_vs

#undef fromInputToSphere
#undef fromSphereToOutput
#undef inverseRadialPixelsTransform
#undef inverseDistortionMetersTransform

#undef fromOutputToSphere
#undef fromSphereToInput
#undef distortionMetersTransform
#undef distortionPixelsTransform

#undef FUNCTION_NAME_4
#undef FUNCTION_NAME_3

// -------------------------- Geo Transform --------------------------

GeoTransform::GeoTransform()
    : fromOutputToSphereHostFn(nullptr),
      fromSphereToInputHostFn(nullptr),
      fromSphereToOutputHostFn(nullptr),
      fromInputToSphereHostFn(nullptr),
      isWithinHostFn(nullptr),
      distortionTransformMetersHostFn(nullptr),
      distortionTransformPixelsHostFn(nullptr),
      inverseDistortionTransformMetersHostFn(nullptr),
      inverseDistortionTransformPixelsHostFn(nullptr),
      panoScale({0.0f, 0.0f}),
      rigSphereRadius(1.0f) {}

GeoTransform::GeoTransform(const GeoTransform& other)
    : fromOutputToSphereHostFn(other.fromOutputToSphereHostFn),
      fromSphereToInputHostFn(other.fromSphereToInputHostFn),
      fromSphereToOutputHostFn(other.fromSphereToOutputHostFn),
      fromInputToSphereHostFn(other.fromInputToSphereHostFn),
      isWithinHostFn(other.isWithinHostFn),
      distortionTransformMetersHostFn(other.distortionTransformMetersHostFn),
      distortionTransformPixelsHostFn(other.distortionTransformPixelsHostFn),
      inverseDistortionTransformMetersHostFn(other.inverseDistortionTransformMetersHostFn),
      inverseDistortionTransformPixelsHostFn(other.inverseDistortionTransformPixelsHostFn),
      panoScale(other.panoScale),
      rigSphereRadius(other.rigSphereRadius) {}

GeoTransform::~GeoTransform() {}

GeoTransform* GeoTransform::create(const PanoDefinition& pano, const InputDefinition& im) {
  std::unique_ptr<GeoTransform> res(new GeoTransform());

  switch (im.getFormat()) {
    case InputDefinition::Format::Rectilinear:
      res->fromSphereToInputHostFn = TransformStack::SphereToRect;
      res->isWithinHostFn = TransformStack::isWithinCropRect;
      res->fromInputToSphereHostFn = TransformStack::RectToSphere;
      break;
    case InputDefinition::Format::Equirectangular:
      res->fromSphereToInputHostFn = TransformStack::SphereToErect;
      res->isWithinHostFn = TransformStack::isWithinCropRect;
      res->fromInputToSphereHostFn = TransformStack::ErectToSphere;
      break;
    case InputDefinition::Format::CircularFisheye:
      res->fromSphereToInputHostFn = TransformStack::SphereToFisheye;
      res->isWithinHostFn = TransformStack::isWithinCropCircle;
      res->fromInputToSphereHostFn = TransformStack::FisheyeToSphere;
      break;
    case InputDefinition::Format::FullFrameFisheye:
      res->fromSphereToInputHostFn = TransformStack::SphereToFisheye;
      res->isWithinHostFn = TransformStack::isWithinCropRect;
      res->fromInputToSphereHostFn = TransformStack::FisheyeToSphere;
      break;
    case InputDefinition::Format::CircularFisheye_Opt:
      res->fromSphereToInputHostFn = TransformStack::SphereToExternal;
      res->isWithinHostFn = TransformStack::isWithinCropCircle;
      res->fromInputToSphereHostFn = TransformStack::ExternalToSphere;
      break;
    case InputDefinition::Format::FullFrameFisheye_Opt:
      res->fromSphereToInputHostFn = TransformStack::SphereToExternal;
      res->isWithinHostFn = TransformStack::isWithinCropRect;
      res->fromInputToSphereHostFn = TransformStack::ExternalToSphere;
      break;
  }

  if (im.getUseMeterDistortion()) {
    res->distortionTransformMetersHostFn = TransformStack::distortionScaled;
    res->distortionTransformPixelsHostFn = TransformStack::noopDistortionTransform;
    res->inverseDistortionTransformMetersHostFn = TransformStack::inverseDistortionScaled;
    res->inverseDistortionTransformPixelsHostFn = TransformStack::noopDistortionTransform;
  } else {
    res->distortionTransformMetersHostFn = TransformStack::noopDistortionTransform;
    res->distortionTransformPixelsHostFn = TransformStack::distortionScaled;
    res->inverseDistortionTransformMetersHostFn = TransformStack::noopDistortionTransform;
    res->inverseDistortionTransformPixelsHostFn = TransformStack::inverseDistortionScaled;
  }

  switch (pano.getProjection()) {
    case PanoProjection::Rectilinear:
      res->fromOutputToSphereHostFn = TransformStack::RectToSphere;
      res->fromSphereToOutputHostFn = TransformStack::SphereToRect;
      break;
    case PanoProjection::Cylindrical:
      return NULL;
    case PanoProjection::Equirectangular:
      res->fromOutputToSphereHostFn = TransformStack::ErectToSphere;
      res->fromSphereToOutputHostFn = TransformStack::SphereToErect;
      break;
    case PanoProjection::FullFrameFisheye:
      res->fromOutputToSphereHostFn = TransformStack::FisheyeToSphere;
      res->fromSphereToOutputHostFn = TransformStack::SphereToFisheye;
      break;
    case PanoProjection::CircularFisheye:
      res->fromOutputToSphereHostFn = TransformStack::FisheyeToSphere;
      res->fromSphereToOutputHostFn = TransformStack::SphereToFisheye;
      break;
    case PanoProjection::Stereographic:
      res->fromOutputToSphereHostFn = TransformStack::StereoToSphere;
      res->fromSphereToOutputHostFn = TransformStack::SphereToStereo;
      break;
    case PanoProjection::Cubemap:
    case PanoProjection::EquiangularCubemap:
      assert(false);
      break;
  }

  res->panoScale.x = TransformGeoParams::computePanoScale(pano.getProjection(), pano.getWidth(), (float)pano.getHFOV());
  res->panoScale.y = TransformGeoParams::computePanoScale(pano.getProjection(), pano.getWidth(), (float)pano.getHFOV());

  res->rigSphereRadius = (float)pano.getSphereScale();

  return res.release();
}

SphericalCoords3 GeoTransform::mapPanoramaToRigSpherical(CenterCoords2 uv) const {
  // TODO: share this code with GPU stack
  /** Coordinates are in pixels, transform to the unit space (eg. in radians) */
  float2 uvScaled = uv.toFloat2() / panoScale;
  /** From panorama to unit-sphere */
  const float3 coords = fromOutputToSphereHostFn(uvScaled);
  return SphericalCoords3(coords);
}

CenterCoords2 GeoTransform::mapPanoramaToInput(const InputDefinition& im, const CenterCoords2 uv,
                                               const int time) const {
  const GeometryDefinition geometry = im.getGeometries().at(time);
  const TransformGeoParams params(im, geometry, rigSphereRadius);
  const float2 coords =
      mapPanoramaToInput(uv.toFloat2(), panoScale, params.getPose(),
                         {(float)geometry.getHorizontalFocal(), (float)geometry.getVerticalFocal()},
                         params.getDistortion(), {(float)im.getCenterX(geometry), (float)im.getCenterY(geometry)});
  return CenterCoords2{coords};
}

CenterCoords2 GeoTransform::mapInputToPanorama(const InputDefinition& im, const CenterCoords2 uv,
                                               const int time) const {
  const GeometryDefinition geometry = im.getGeometries().at(time);
  const TransformGeoParams params(im, geometry, rigSphereRadius);
  const float2 uvt =
      mapInputToPanorama(uv.toFloat2(), panoScale, params.getPoseInverse(), rigSphereRadius,
                         {(float)geometry.getHorizontalFocal(), (float)geometry.getVerticalFocal()},
                         params.getDistortion(), {(float)im.getCenterX(geometry), (float)im.getCenterY(geometry)});
  return CenterCoords2{uvt};
}

CenterCoords2 GeoTransform::mapRigSphericalToInput(const InputDefinition& im, const SphericalCoords3& point,
                                                   const int time) const {
  const GeometryDefinition geometry = im.getGeometries().at(time);
  const TransformGeoParams params(im, geometry, 1.0f);
  const float2 uv = mapRigSphericalToInput(
      point.toFloat3(), params.getPose(), {(float)geometry.getHorizontalFocal(), (float)geometry.getVerticalFocal()},
      params.getDistortion(), {(float)im.getCenterX(geometry), (float)im.getCenterY(geometry)});
  return CenterCoords2{uv};
}

SphericalCoords3 GeoTransform::mapInputToRigSpherical(const InputDefinition& im, const CenterCoords2 uv, const int time,
                                                      const float rigSphereRadius) const {
  const GeometryDefinition geometry = im.getGeometries().at(time);
  const TransformGeoParams params(im, geometry, rigSphereRadius);

  const float3 ptRig =
      mapInputToRigSpherical(uv.toFloat2(), params.getPoseInverse(), rigSphereRadius,
                             {(float)geometry.getHorizontalFocal(), (float)geometry.getVerticalFocal()},
                             params.getDistortion(), {(float)im.getCenterX(geometry), (float)im.getCenterY(geometry)});
  return SphericalCoords3{ptRig};
}

SphericalCoords3 GeoTransform::mapInputToScaledCameraSphereInRigBase(const InputDefinition& im, const CenterCoords2 uv,
                                                                     const int time, float cameraSphereRadius) const {
  const GeometryDefinition geometry = im.getGeometries().at(time);
  const TransformGeoParams params(im, geometry, 1.0f);

  float3 pt =
      mapInputToCameraSphere(uv.toFloat2(), {(float)geometry.getHorizontalFocal(), (float)geometry.getVerticalFocal()},
                             params.getDistortion(), {(float)im.getCenterX(geometry), (float)im.getCenterY(geometry)});
  pt *= cameraSphereRadius;

  // Transform camera to rig
  pt = transformSphere(pt, params.getPoseInverse());
  return SphericalCoords3{pt};
}

float GeoTransform::computeInputMinimumRigSphereRadius(const InputDefinition& im, int time) const {
  const GeometryDefinition geometry = im.getGeometries().at(time);
  const TransformGeoParams params(im, geometry, 1.0f);

  /* Camera center in spherical (rig) space */
  const float3 camcenterpt = transformSphere(make_float3(0.f, 0.f, 0.f), params.getPoseInverse());

  /* return norm of camera center in spherical (rig) space */
  return length(camcenterpt);
}

bool GeoTransform::isWithinInputBounds(const InputDefinition& im, TopLeftCoords2 uv) const {
  return isWithinHostFn(uv.toFloat2(), (float)im.getWidth(), (float)im.getHeight(), (float)im.getCropLeft(),
                        (float)im.getCropRight(), (float)im.getCropTop(), (float)im.getCropBottom());
}

}  // namespace TransformStack
}  // namespace Core
}  // namespace VideoStitch
