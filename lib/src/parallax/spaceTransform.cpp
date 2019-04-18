// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "spaceTransform.hpp"

#include "backend/common/vectorOps.hpp"
#include "core/geoTransform.hpp"

#include "libvideostitch/inputDef.hpp"

#include <cstdlib>
#include <memory>

namespace VideoStitch {
namespace Core {

Vector3<double> SpaceTransform::getAverageSphericalCoord(const Core::PanoDefinition& pano,
                                                         const Core::InputDefinition& im, const int sampleCount) {
  // Create transformation from both inputs
  std::unique_ptr<TransformStack::GeoTransform> geoTransform(TransformStack::GeoTransform::create(pano, im));

  const int width = (int)im.getWidth();
  const int height = (int)im.getHeight();
  std::srand(0);  // Use 0 as seed for deterministic random process
  float3 avgCoord = make_float3(0, 0, 0);
  // Generate a random subset of input, map it to the sphere and take the average direction
  for (int i = 0; i < sampleCount; i++) {
    const Core::CenterCoords2 uv((float)(std::rand() % width - width / 2), (float)(std::rand() % height - height / 2));
    avgCoord += geoTransform->mapInputToScaledCameraSphereInRigBase(im, uv, 0).toFloat3();
  }
  avgCoord /= (float)sampleCount;
  return Vector3<double>(avgCoord.x, avgCoord.y, avgCoord.z);
}

SpaceTransform::SpaceTransform(const Vector3<double>& oldPole, const Vector3<double>& newPole) {
  VideoStitch::Quaternion<double> q = Quaternion<double>::fromTwoVectors(oldPole, newPole);
  // This is the rotation matrix to rotate image around certain pole

  /*Set rotation values*/
  Matrix33<double> poleRotationMatrix = q.toRotationMatrix();
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      pose.values[row][col] = (float)poleRotationMatrix(row, col);
    }
  }
  pose.values[0][3] = 0;
  pose.values[1][3] = 0;
  pose.values[2][3] = 0;

  /*Set inverse values*/
  Matrix33<double> poleRotationMatrixInverse;
  poleRotationMatrix.inverse(poleRotationMatrixInverse);
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      poseInverse.values[row][col] = (float)poleRotationMatrixInverse(row, col);
    }
  }
  poseInverse.values[0][3] = 0;
  poseInverse.values[1][3] = 0;
  poseInverse.values[2][3] = 0;
}

}  // namespace Core
}  // namespace VideoStitch
