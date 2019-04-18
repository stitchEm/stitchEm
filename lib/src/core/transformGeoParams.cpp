// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "transformGeoParams.hpp"

#include "radial.hpp"

#include "common/angles.hpp"

#include "libvideostitch/geometryDef.hpp"

#include <cassert>

namespace VideoStitch {
namespace Core {

TransformGeoParams::TransformGeoParams(const InputDefinition& im, const GeometryDefinition& geometry,
                                       const PanoDefinition& pano)
    : TransformGeoParams(im, geometry, pano.getSphereScale()) {}

TransformGeoParams::TransformGeoParams(const InputDefinition& im, const GeometryDefinition& geometry,
                                       const double sphereScale) {
  const double yaw = geometry.getYaw();
  const double pitch = geometry.getPitch();
  const double roll = geometry.getRoll();

  /*Add radial correction*/
  computeRadialParams(im, geometry, distortion.values[0], distortion.values[1], distortion.values[2],
                      distortion.values[3], distortion.values[4]);

  /*Add non-radial correction*/
  distortion.values[5] = float(geometry.getDistortP1());
  distortion.values[6] = float(geometry.getDistortP2());
  distortion.values[7] = float(geometry.getDistortS1());
  distortion.values[8] = float(geometry.getDistortS2());
  distortion.values[9] = float(geometry.getDistortS3());
  distortion.values[10] = float(geometry.getDistortS4());
  const float tauX = float(geometry.getDistortTau1());
  const float tauY = float(geometry.getDistortTau2());

  /*Prepare flags to signal enabled distortion values*/
  distortion.distortionBitFlag = 0;
  if (distortion.values[5] != 0.0f || distortion.values[6] != 0.0f) {
    distortion.distortionBitFlag |= TANGENTIAL_DISTORTION_BIT;
  }
  if (distortion.values[7] != 0.0f || distortion.values[8] != 0.0f || distortion.values[9] != 0.0f ||
      distortion.values[10] != 0.0f) {
    distortion.distortionBitFlag |= THIN_PRISM_DISTORTION_BIT;
  }
  if (tauX != 0.0f || tauY != 0.0f) {
    distortion.distortionBitFlag |= SCHEIMPFLUG_DISTORTION_BIT;
  }

  /*Prepare forward and inverse Scheimpflug transform matrices, constant for this given geometry*/
  if (distortion.distortionBitFlag & SCHEIMPFLUG_DISTORTION_BIT) {
    const float cTauX = std::cos(tauX);
    const float sTauX = std::sin(tauX);
    const float cTauY = std::cos(tauY);
    const float sTauY = std::sin(tauY);

    const Matrix33<float> matRotXY(cTauY, sTauY * sTauX, -sTauY * cTauX, 0.0f, cTauX, sTauX, sTauY, -cTauY * sTauX,
                                   cTauY * cTauX);
    const Matrix33<float> matProjZ(matRotXY(2, 2), 0.0f, -matRotXY(0, 2), 0.0f, matRotXY(2, 2), -matRotXY(1, 2), 0.0f,
                                   0.0f, 1.0f);
    const Matrix33<float> forward(matProjZ * matRotXY);
    std::memcpy(distortion.scheimpflugForward.values, forward.data(), sizeof(distortion.scheimpflugForward.values));

    if (std::abs(matRotXY(2, 2)) > 1.e-6f) {
      const float inv = 1.f / matRotXY(2, 2);
      const Matrix33<float> invMatProjZ(inv, 0.f, inv * matRotXY(0, 2), 0.0f, inv, inv * matRotXY(1, 2), 0.0f, 0.0f,
                                        1.0f);
      const Matrix33<float> backward(matRotXY.transpose() * invMatProjZ);
      std::memcpy(distortion.scheimpflugInverse.values, backward.data(), sizeof(distortion.scheimpflugInverse.values));
    } else {
      /*Should not get here*/
      assert(false);
    }
  }

  // construct pose rig -> camera, 3x4 [R | t]
  Matrix33<double> mat = Matrix33<double>::fromEulerZXY(degToRad(yaw), degToRad(pitch), degToRad(roll));

  // Set rotation values
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      pose.values[row][col] = (float)(mat(row, col));
    }
  }

  // Set translation values
  pose.values[0][3] = (float)geometry.getTranslationX();
  pose.values[1][3] = (float)geometry.getTranslationY();
  pose.values[2][3] = (float)geometry.getTranslationZ();

  // construct inverse pose camera -> rig
  // T^-1 = [R^t | -R^t * translation]
  for (int row = 0; row < 3; row++) {
    double translation = 0.0;
    for (int col = 0; col < 3; col++) {
      // R^t
      poseInverse.values[row][col] = pose.values[col][row];
      // R^t * translation
      translation -= poseInverse.values[row][col] * pose.values[col][3];
    }
    poseInverse.values[row][3] = (float)translation;
  }

  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      // scale rig sphere
      pose.values[row][col] *= (float)sphereScale;

      // scale camera sphere, equivalent to scaling rig sphere if `translation == 0`
      //
      // Note: this scales a point on a sphere in camera-centric coordinates.
      // If there is translation the resulting point in rig coordinates will
      // not have `length(pt) == sphereScale`! Additional computations will be required.
      poseInverse.values[row][col] *= (float)sphereScale;
    }
  }
}

float TransformGeoParams::computePanoScale(const PanoProjection& proj, const int64_t panoWidth,
                                           const float panoHFOVDeg) {
  const float panoHFOVRad = (float)degToRad(panoHFOVDeg);
  switch (proj) {
    case PanoProjection::Rectilinear:
      return (float)panoWidth / (2.0f * tanf(panoHFOVRad / 2.0f));
    case PanoProjection::Stereographic:
      return (float)panoWidth / (4.0f * tanf(panoHFOVRad / 4.0f));
    case PanoProjection::Cylindrical:
    case PanoProjection::Equirectangular:
    case PanoProjection::FullFrameFisheye:
    case PanoProjection::CircularFisheye:
      return (float)panoWidth / panoHFOVRad;
    case PanoProjection::Cubemap:
    case PanoProjection::EquiangularCubemap:
      assert(false);
      return 0.;
  }
  assert(false);
  return 0.0f;
}

double TransformGeoParams::getInverseDemiDiagonalSquared(const InputDefinition& im) {
  return im.hasCroppedArea() ? 4.0f / (float)(im.getCroppedWidth() * im.getCroppedWidth() +
                                              im.getCroppedHeight() * im.getCroppedHeight())
                             : 4.0f / (float)(im.getWidth() * im.getWidth() + im.getHeight() * im.getHeight());
}

float TransformGeoParams::computeInputScale(InputDefinition::Format imFmt, int64_t imWidth, float imHFOVDeg) {
  const float imHFOVRad = (float)degToRad(imHFOVDeg);
  switch (imFmt) {
    case InputDefinition::Format::Rectilinear:
      return (float)imWidth / (2.0f * tanf(imHFOVRad / 2.0f));
    case InputDefinition::Format::Equirectangular:
    case InputDefinition::Format::CircularFisheye:
    case InputDefinition::Format::FullFrameFisheye:
      return (float)imWidth / imHFOVRad;
    case InputDefinition::Format::CircularFisheye_Opt:
    case InputDefinition::Format::FullFrameFisheye_Opt:
      const float fov2 = imHFOVRad / 2.0f;
      return (float)((double)imWidth * (cos(fov2) + 1.0) / (2.0 * sin(fov2)));
  }
  return 0.0f;
}

double TransformGeoParams::computeFovFromFocal(InputDefinition::Format imFmt, int64_t imWidth, double focal) {
  double fovrad = 0.0;

  switch (imFmt) {
    case InputDefinition::Format::Rectilinear:
      fovrad = 2.0 * atan((double)imWidth / (2.0 * focal));
      break;
    case InputDefinition::Format::Equirectangular:
    case InputDefinition::Format::CircularFisheye:
    case InputDefinition::Format::FullFrameFisheye:
      fovrad = (double)imWidth / focal;
      break;
    case InputDefinition::Format::CircularFisheye_Opt:
    case InputDefinition::Format::FullFrameFisheye_Opt:
      if (focal > 1e-6) {
        fovrad = 4 * atan((double)imWidth / (2.0 * focal));
      }
      break;
  }

  return radToDeg(fovrad);
}

double TransformGeoParams::computeFov(const InputDefinition& im, double focal) {
  if (im.hasCroppedArea()) {
    return computeFovFromFocal(im.getFormat(), im.getCroppedWidth(), (float)focal);
  } else {
    return computeFovFromFocal(im.getFormat(), im.getWidth(), (float)focal);
  }
}

float TransformGeoParams::computeHorizontalScale(const InputDefinition& im, double fov) {
  if (im.hasCroppedArea()) {
    return computeInputScale(im.getFormat(), im.getCroppedWidth(), (float)fov);
  } else {
    return computeInputScale(im.getFormat(), im.getWidth(), (float)fov);
  }
}

float TransformGeoParams::computeVerticalScale(const InputDefinition& im, double fov) {
  if (im.hasCroppedArea()) {
    return computeInputScale(im.getFormat(), im.getCroppedHeight(), (float)fov);
  } else {
    return computeInputScale(im.getFormat(), im.getHeight(), (float)fov);
  }
}

}  // namespace Core
}  // namespace VideoStitch
