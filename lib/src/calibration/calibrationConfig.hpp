// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/controlPointListDef.hpp"

#include <opencv2/core/core.hpp>

#include <memory>

namespace VideoStitch {

namespace Core {
class RigDefinition;
}

namespace Calibration {

class CalibrationConfig {
 public:
  explicit CalibrationConfig(const Ptv::Value* config);
  ~CalibrationConfig() = default;

  CalibrationConfig(const CalibrationConfig&);

  bool isValid() const { return isConfigValid; }

  bool isApplyingPresetsOnly() const { return applyPresetsOnly; }

  bool isInDeshuffleMode() const { return deshuffleMode; }

  bool isInDeshuffleModeOnly() const { return deshuffleModeOnly; }

  bool isInDeshufflePreserveReadersOrder() const { return deshuffleModePreserveReadersOrder; }

  bool isFovDefined() const { return !automaticFovIterate; }

  bool hasSingleFocal() const { return singleFocalForAllLenses; }

  bool isInImproveMode() const { return improveMode; }

  bool isSavingDebugSnapshots() const { return dumpDebugSnapshots; }

  void setIsSavingDebugSnapshots(const bool save) { dumpDebugSnapshots = save; }

  bool isGeneratingSyntheticKeypoints() const { return useSyntheticKeypoints; }

  void setIsGeneratingSyntheticKeypoints(const bool generate) { useSyntheticKeypoints = generate; }

  double getSyntheticKeypointsGridWidth() const { return syntheticKeypointsGridWidth; }

  double getSyntheticKeypointsGridHeight() const { return syntheticKeypointsGridHeight; }

  std::vector<unsigned int> getFrames() const { return frames; }

  const std::shared_ptr<Core::RigDefinition> getRigPreset() const { return rig; }

  double getInitialHFov() const { return initialHFovValue; }

  /**
  @brief Get Mask for a given camera/input
  @details This mask will define if a point is allowed to be detected here
  @param cam_id the camera id
  @param frame_id the frame_id
  @param mask the output mask
  @return true if the mask exist
  */
  bool getControlPointsMask(cv::Mat& mask, size_t cam_id, size_t frame_id) const {
    std::map<std::pair<size_t, size_t>, cv::Mat>::const_iterator it;
    std::pair<size_t, size_t> key;
    key.first = (size_t)cam_id;
    key.second = (size_t)frame_id;
    it = masksmap.find(key);
    if (it == masksmap.end()) {
      return false;
    }

    mask = it->second;

    return true;
  }

  const Core::ControlPointList& getControlPointList() const { return cpList; }

  void setRigDefinition(const std::shared_ptr<Core::RigDefinition>& myRig) { rig = myRig; }

  /* Control Point extractor */

  std::string getExtractorName() const { return extractor; }

  std::string getMatcherName() const { return matcher; }

  int getOctaves() const { return octaves; }

  int getSubLevels() const { return sublevels; }

  double getDetectionThreshold() const { return threshold; }

  double getNNDRRatio() const { return nndr_ratio; }

  /* Control Point filter */

  double getAngleThreshold() const { return angle_threshold; }

  double getMinRatioInliers() const { return min_ratio_inliers; }

  int getMinSamplesForFit() const { return min_samples_for_fit; }

  double getRatioOutliers() const { return ratio_outliers; }

  double getProbaDrawOutlierFree() const { return proba_draw_outlier_free; }

  double getDecimatingGridSize() const { return decimating_grid_size; }

 private:
  bool isConfigValid;

  /* Calibration */

  /* Apply rig / camera presets without calculating control points*/
  bool applyPresetsOnly;

  /* Reorders inputs as part of the algorithm */
  bool deshuffleMode;

  /* Applies deshuffle algorithm only */
  bool deshuffleModeOnly;

  /* Applies deshuffle algorithm keeping the inputs indexes as the original ones*/
  bool deshuffleModePreserveReadersOrder;

  /* Calculates the HFOV value */
  bool automaticFovIterate;

  /* Initial HFOV value */
  double initialHFovValue;
  bool singleFocalForAllLenses;
  bool improveMode;
  bool dumpDebugSnapshots;
  bool useSyntheticKeypoints;
  double syntheticKeypointsGridWidth;
  double syntheticKeypointsGridHeight;

  /* List of frames */
  std::vector<unsigned int> frames;

  /* Map indexed by camera id and frame id to store masks for point detection*/
  std::map<std::pair<size_t, size_t>, cv::Mat> masksmap;

  /* Rig preset */
  std::shared_ptr<Core::RigDefinition> rig;

  /* Calibration ControlPoints */
  Core::ControlPointList cpList;

  /* Control Point extractor */
  std::string extractor;
  std::string matcher;
  int octaves;
  int sublevels;
  double threshold;
  double nndr_ratio;

  /* Control Point filter */
  double angle_threshold;
  double min_ratio_inliers;
  int min_samples_for_fit;
  double ratio_outliers;
  double proba_draw_outlier_free;
  double decimating_grid_size;
};

}  // namespace Calibration
}  // namespace VideoStitch
