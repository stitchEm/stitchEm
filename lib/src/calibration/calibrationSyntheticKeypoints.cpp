// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibration.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/logging.hpp"

#include <core/geoTransform.hpp>

#include <set>

namespace VideoStitch {
namespace Calibration {

/**
 * Uses the project geometry in PanoDefinition to generate artificial keypoints covering all inputs
 * and places them in syntheticmatches_map
 * syntheticmatches_map is used by the calibration algorithm to have keypoints connecting all inputs
 * on areas where no real keypoint could be detected and matched
 */
Status Calibration::generateSyntheticControlPoints(const Core::PanoDefinition& pano) {
  syntheticmatches_map.clear();

  // prepare transforms to check for picture crop areas
  std::vector<std::unique_ptr<Core::TransformStack::GeoTransform>> transforms;
  std::vector<Core::TopLeftCoords2> inputCenters;
  std::set<std::tuple<double, double, double>> setOfInputRotations;

  for (const auto& videoInputDef : pano.getVideoInputs()) {
    // GeoTransform
    transforms.push_back(std::unique_ptr<Core::TransformStack::GeoTransform>(
        Core::TransformStack::GeoTransform::create(pano, videoInputDef)));
    // Input centers
    inputCenters.push_back(Core::TopLeftCoords2((float)(videoInputDef.get().getWidth() / 2),
                                                (float)(videoInputDef.get().getHeight() / 2)));
    // input rotations
    const Core::GeometryDefinition& inputGeometry = videoInputDef.get().getGeometries().at(0);
    setOfInputRotations.insert(
        std::make_tuple(inputGeometry.getYaw(), inputGeometry.getPitch(), inputGeometry.getRoll()));
  }

  // check that each input has a different rotation
  if (setOfInputRotations.size() != (size_t)pano.numVideoInputs()) {
    return {Origin::CalibrationAlgorithm, ErrType::InvalidConfiguration,
            "Cannot generate synthetic keypoints for calibration, not all inputs have distinct geometries"};
  }

  for (videoreaderid_t camid = 0; camid < pano.numVideoInputs(); ++camid) {
    int64_t width = pano.getVideoInput(camid).getWidth();
    int64_t height = pano.getVideoInput(camid).getHeight();
    // distribute 2D points evenly on the input
    for (double y = 0; y < height; y += height / calibConfig.getSyntheticKeypointsGridHeight()) {
      for (double x = 0; x < width; x += width / calibConfig.getSyntheticKeypointsGridWidth()) {
        Core::TopLeftCoords2 point2D((float)x, (float)y);

        // check that point is within crop area
        if (transforms[camid]->isWithinInputBounds(pano.getVideoInput(camid), point2D)) {
          Core::CenterCoords2 centerPointer2D(point2D, inputCenters[camid]);
          Core::SphericalCoords3 scaledPoint3D = transforms[camid]->mapInputToRigSpherical(
              pano.getVideoInput(camid), centerPointer2D, 0, (float)pano.getSphereScale());

          // check that the sphere point is at the right sphereScale
          assert(std::abs(std::sqrt(scaledPoint3D.x * scaledPoint3D.x + scaledPoint3D.y * scaledPoint3D.y +
                                    scaledPoint3D.z * scaledPoint3D.z) -
                          pano.getSphereScale()) < 1e-3);

#ifndef NDEBUG
          // check that reprojection is bijective
          Core::CenterCoords2 reprojected =
              transforms[camid]->mapRigSphericalToInput(pano.getVideoInput(camid), scaledPoint3D, 0);

          assert(std::abs(centerPointer2D.x - reprojected.x) < 1e-3f &&
                 std::abs(centerPointer2D.y - reprojected.y) < 1e-3f && "reprojection failed");
#endif

          // project scaledPoint3D on other inputs
          for (videoreaderid_t othercamid = 0; othercamid < pano.numVideoInputs(); ++othercamid) {
            if (camid == othercamid) {
              continue;
            }

            Core::CenterCoords2 centerProjected =
                transforms[othercamid]->mapRigSphericalToInput(pano.getVideoInput(othercamid), scaledPoint3D, 0);
            Core::TopLeftCoords2 topLeftProjected(centerProjected, inputCenters[othercamid]);

            // is within crop area ?
            if (transforms[othercamid]->isWithinInputBounds(pano.getVideoInput(othercamid), topLeftProjected)) {
              // add the ControlPoint with a high score, to give it less priority if a real ControlPoint is available
              if (camid < othercamid) {
                Core::ControlPoint cp(
                    camid, othercamid, x, y, topLeftProjected.x, topLeftProjected.y, -1 /* frameNumber */, 0.,
                    std::numeric_limits<
                        double>::max() /* max score so that they get less priority than real keypoints */,
                    true /* artificial */);
                syntheticmatches_map[{camid, othercamid}].push_back(cp);
              } else {
                Core::ControlPoint cp(
                    othercamid, camid, topLeftProjected.x, topLeftProjected.y, x, y, -1 /* frameNumber */, 0.,
                    std::numeric_limits<
                        double>::max() /* max score so that they get less priority than real keypoints */,
                    true /* artificial */);
                syntheticmatches_map[{othercamid, camid}].push_back(cp);
              }
            }
          }
        }
      }
    }
  }

  if (!syntheticmatches_map.empty()) {
    Logger::get(Logger::Verbose) << "Calibration: generated synthetic keypoints" << std::endl;
    for (const auto& it : syntheticmatches_map) {
      std::stringstream message;
      message << "  Inputs " << it.first.first << " and " << it.first.second << ": " << it.second.size() << std::endl;
      Logger::get(Logger::Verbose) << message.str() << std::flush;
    }
  }

  return Status::OK();
}

}  // namespace Calibration
}  // namespace VideoStitch
