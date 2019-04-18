// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/coordinates.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/controlPointListDef.hpp"

namespace VideoStitch {
namespace EpipolarCurves {

/**
 * @brief Configuration used by the EpipolarCurvesAlgorithm
 */
class VS_EXPORT EpipolarCurvesConfig {
 public:
  explicit EpipolarCurvesConfig(const Ptv::Value* config);
  ~EpipolarCurvesConfig() = default;
  EpipolarCurvesConfig(const EpipolarCurvesConfig&);

  bool isValid() const { return isConfigValid; }

  bool getIsAutoPointMatching() const { return autoPointMatching; }

  double getDecimationCellFactor() const { return decimationCellFactor; }

  double getSphericalGridRadius() const { return sphericalGridRadius; }

  double getImageMaxOutputDepth() const { return imageMaxOutputDepth; }

  std::vector<frameid_t> getFrames() const { return frames; }

  std::map<videoreaderid_t, std::vector<Core::TopLeftCoords2>> getSinglePointsMap() const { return singlePointsMap; }

  std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList> getMatchedPointsMap() const {
    return matchedPointsMap;
  }

 private:
  bool isConfigValid;

  /* Auto selection of matched points */
  bool autoPointMatching;
  double decimationCellFactor;
  double sphericalGridRadius;
  double imageMaxOutputDepth;

  /* List of frames */
  std::vector<frameid_t> frames;

  /* List of points to show in input pictures */
  std::map<videoreaderid_t, std::vector<Core::TopLeftCoords2>> singlePointsMap;

  /* List of matched points to show in input pictures */
  std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList> matchedPointsMap;
};

}  // namespace EpipolarCurves
}  // namespace VideoStitch
