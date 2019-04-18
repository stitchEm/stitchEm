// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __CALIBRATION_HPP__
#define __CALIBRATION_HPP__

#include "calibrationConfig.hpp"
#include "calibrationProgress.hpp"
#include "cvImage.hpp"

#include "libvideostitch/controlPointListDef.hpp"

#include <memory>
#include <vector>
#include <random>
#include <sstream>
#include <set>

namespace VideoStitch {

namespace Testing {
void testKeypointsConnectivity();
}

namespace Calibration {

class Camera;

/**
 * @brief This algorithm calibrates the panoramic multi-camera system based on the overlap between input images.
 * @details The multi-camera system is assumed to be central, i.e. the cameras should approximately share a common
 * optical center.
 * @details In practice a small distance (a few centimeters) between optical centers is good enough but will introduce
 * ghosting effects
 * @details on near objects due to parallax.
 */
class VS_EXPORT Calibration : public Util::OpaquePtr {
 public:
  Calibration(const CalibrationConfig& config, CalibrationProgress& progress);
  virtual ~Calibration();

  /**
  @brief Process current set of images
  @param pano the input output parameters for the panorama
  @param rig the input set of images
  */
  Status process(Core::PanoDefinition& pano, const RigCvImages& rig);

 protected:
  static std::vector<double> getFOVValues();
  void setupRig();
  Status extractControlPoints();
  Status generateSyntheticControlPoints(const Core::PanoDefinition& pano);
  Status estimateExtrinsics(double& cost);
  Status optimize(Core::PanoDefinition& pano);
  void filterFromPresets(Core::ControlPointList& filtered, const Core::ControlPointList& input, videoreaderid_t idcam1,
                         videoreaderid_t idcam2, const double sphereScale);
  Status filterControlPoints(const double sphereScale, double* const cost = nullptr);
  void fillPanoWithControlPoints(Core::PanoDefinition& pano);
  void projectFromCurrentSettings(Core::ControlPointList& list, videoreaderid_t idcam1, videoreaderid_t idcam2,
                                  const double sphereScale);

  /**
   @brief Extract the calibration keypoints (used for FOV iterations)
   @param rig the input set of images
   */
  Status extractAndMatchControlPoints(const RigCvImages& rig);

  /**
   @brief Process current extracted keypoints (used for FOV iterations)
   @param pano the input/output parameters for the panorama
   @param rig the input set of images
   @param fullOptimization performs full parameters refinement (otherwise, stops at the computation of extrinsics
   parameters)
   */
  Status processGivenMatchedControlPoints(Core::PanoDefinition& pano, const bool fullOptimization);

  /**
   @brief Just apply the presets geometry without running the actual calibration
   @param pano the input/output panorama definition
   */
  Status applyPresetsGeometry(Core::PanoDefinition& pano);

  /**
   @brief Updates the calibration config (used for FOV iterations)
   @param config calibration config
   */
  void updateCalibrationConfig(const CalibrationConfig& config);

  /**
   @brief Deshuffle the inputs from the rig presets
   */
  Status deshuffleInputs(Core::PanoDefinition& pano);

  /**
   @brief Applies the given permutation to the readers of panorama definition (not permuting the inputs), the
   machedpoints_map and the rig image
   @param pano panorama definition
   @param preserveReadersOrder if true, preserves the readers order (otherwise, applies permutation to the readers only)
   @param permutation permutation to apply.
   @return true if permutation could be applied, else false
   */
  bool applyPermutation(Core::PanoDefinition* pano, const bool preserveReadersOrder,
                        const std::vector<videoreaderid_t>& permutation);

  /**
   * @brief Iterates over a list of possible HFOV values and finds the best for the current calibration.
   * @param pano The panorama
   * @return Ok if the HFOV is found and applied.
   */
  Status calculateFOV(Core::PanoDefinition& pano);

  /**
   * @brief analyses the connectivity of the keypoints map
   * @param keypoints_map keypoints map
   * @param numCameras number of cameras
   * @param reportStringStream optional stringstream to receive report messages
   * @param connectivity optional map of connectivity of inputs
   * @param singleConnectedInputs optional set of single connected inputs
   * @param nonConnectedInputs optional set of non connected inputs
   * @return false if one or several inputs are not connected to any other input
   */
  static bool analyzeKeypointsConnectivity(
      const std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList>& keypoints_map,
      const videoreaderid_t numCameras, std::stringstream* reportStringStream = nullptr,
      std::map<videoreaderid_t, std::set<videoreaderid_t>>* connectivity = nullptr,
      std::set<videoreaderid_t>* singleConnectedInputs = nullptr,
      std::set<videoreaderid_t>* nonConnectedInputs = nullptr);

  CalibrationConfig calibConfig;
  CalibrationProgress& progress;
  std::vector<std::shared_ptr<Camera>> cameras;
  std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList> matchedpoints_map;
  std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList> configmatches_map;
  std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList> syntheticmatches_map;
  std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList> globalmatches_map;
  std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList> extrinsicsmatches_map;
  RigCvImages rigInputImages;
  std::default_random_engine gen;
  double initialHFOV;

  // friending for testing
  friend void VideoStitch::Testing::testKeypointsConnectivity();
};

}  // namespace Calibration
}  // namespace VideoStitch

#endif
