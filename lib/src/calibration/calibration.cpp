// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __clang_analyzer__  // VSA-7040

#include "calibration.hpp"

#include "camera_fisheye.hpp"
#include "camera_nextfisheye.hpp"
#include "camera_perspective.hpp"
#include "keypointExtractor.hpp"
#include "keypointMatcher.hpp"
#include "controlPointFilter.hpp"
#include "rigGraph.hpp"
#include "rigBuilder.hpp"
#include "calibrationRefinement.hpp"
#include "calibrationUtils.hpp"

#include "libvideostitch/rigDef.hpp"
#include "libvideostitch/rigCameraDef.hpp"
#include "libvideostitch/cameraDef.hpp"
#include "libvideostitch/logging.hpp"
#include <common/angles.hpp>
#include <common/container.hpp>

#include <algorithm>

namespace VideoStitch {
namespace Calibration {

#define CALIBRATION_RANDOM_SEED 42
#define REPORT_STATS_ON_FILTERED_POINTS 0
#define FILTERED_POINTS_WEIGHT_IN_SCORE 1000.

#define FOV_ITERATE_START 70
#define FOV_ITERATE_END 230
#define FOV_ITERATE_INC 5

Calibration::Calibration(const CalibrationConfig& config, CalibrationProgress& progress)
    : calibConfig(config), progress(progress), gen(CALIBRATION_RANDOM_SEED), initialHFOV(0.0) {
  /*Setup initial rig from rigdefinition*/
  setupRig();

  /*Get control points from config object and add them to configmatches_map map*/
  for (auto& cp : config.getControlPointList()) {
    configmatches_map[{cp.index0, cp.index1}].push_back(cp);
  }
}

Calibration::~Calibration() {}

void Calibration::updateCalibrationConfig(const CalibrationConfig& config) {
  calibConfig = config;
  /*Update rig*/
  setupRig();
}

Status Calibration::processGivenMatchedControlPoints(Core::PanoDefinition& pano, const bool fullOptimization) {
  /*If not doing full optimization, disable progress reporting (mechanism used for the FOV iterations)*/
  if (!fullOptimization) {
    progress.disable();
  }

  pano.setHasBeenCalibrationDeshuffled(false);
  if (calibConfig.isInDeshuffleMode()) {
    FAIL_RETURN(deshuffleInputs(pano));
  }

  FAIL_RETURN(filterControlPoints(pano.getSphereScale()));
  double extrinsicsCost;
  FAIL_RETURN(estimateExtrinsics(extrinsicsCost));

  if (fullOptimization) {
    /*At this stage, save the extrinsics control points to the output pano*/
    fillPanoWithControlPoints(pano);
    if (!calibConfig.isInDeshuffleModeOnly()) {
      FAIL_RETURN(optimize(pano));
    }
    Logger::get(Logger::Verbose) << "Calibration finished." << std::endl;
  } else {
    /*Fill in the extrinsics score*/
    pano.setCalibrationCost(extrinsicsCost);
  }

  /*Re-enable progress reporting*/
  if (!fullOptimization) {
    progress.enable();
  }
  return Status::OK();
}

Status Calibration::process(Core::PanoDefinition& pano, const RigCvImages& rig) {
  if (calibConfig.isApplyingPresetsOnly()) {
    /*Just apply the presets*/
    FAIL_RETURN(applyPresetsGeometry(pano));
  } else {
    double initialHFOV = calibConfig.getInitialHFov();
    /*Perform calibration - start by getting control points*/
    FAIL_RETURN(extractAndMatchControlPoints(rig));

    if (!calibConfig.isFovDefined()) {
      FAIL_RETURN(calculateFOV(pano));
    }

    if (calibConfig.isGeneratingSyntheticKeypoints()) {
      FAIL_RETURN(generateSyntheticControlPoints(pano));
    }

    FAIL_RETURN(processGivenMatchedControlPoints(pano, true /* full optimization */));
    pano.setCalibrationInitialHFOV(initialHFOV);
  }
  return Status::OK();
}

std::vector<double> Calibration::getFOVValues() {
  std::vector<double> values;
  for (int i = FOV_ITERATE_START; i <= FOV_ITERATE_END; i += FOV_ITERATE_INC) {
    values.push_back(i);
  }
  return values;
}

void Calibration::setupRig() {
  const std::shared_ptr<Core::RigDefinition> rigdefinition = calibConfig.getRigPreset();
  assert(rigdefinition);

  /*Clear cameras before setting up new ones*/
  cameras.clear();

  /*Set cameras */
  size_t count = rigdefinition->getRigCameraDefinitionCount();
  for (size_t idcam = 0; idcam < count; ++idcam) {
    Core::RigCameraDefinition input;
    rigdefinition->getRigCameraDefinition(input, idcam);
    Camera* camera = nullptr;

    switch (input.getCamera()->getType()) {
      case Core::InputDefinition::Format::Rectilinear:
        camera = new Camera_Perspective;
        break;
      case Core::InputDefinition::Format::CircularFisheye_Opt:
      case Core::InputDefinition::Format::FullFrameFisheye_Opt:
        camera = new Camera_NextFisheye;
        break;
      default:
        camera = new Camera_Fisheye;
        break;
    }

    camera->setFormat(input.getCamera()->getType());
    std::shared_ptr<Camera> pCamera(camera);
    pCamera->setupWithRigCameraDefinition(input);
    cameras.push_back(pCamera);
  }
}

Status Calibration::extractAndMatchControlPoints(const RigCvImages& rig) {
  cv::theRNG().state = CALIBRATION_RANDOM_SEED;
  const size_t numCameras = rig.size();

  /*Set rig pictures*/
  rigInputImages = rig;

  /*Set frame numbers*/
  std::vector<unsigned int> frameNumbers = calibConfig.getFrames();

  /*Extraction and description of features on a list of images*/
  KeypointExtractor kpExtractor(calibConfig.getOctaves(), calibConfig.getSubLevels(),
                                calibConfig.getDetectionThreshold());
  KeypointMatcher kpMatcher(calibConfig.getNNDRRatio());

  /*Loop over all frames of the set used for calibration*/
  for (size_t idinput = 0; idinput < rigInputImages[0].size(); idinput++) {
    std::vector<KPList> keypoints;
    std::vector<DescriptorList> descriptors;
    keypoints.resize(numCameras);
    descriptors.resize(numCameras);

    Logger::get(Logger::Verbose) << "Extracting points from rig #" << idinput << std::endl;

    /*Loop over cameras*/
    for (size_t idcam = 0; idcam < numCameras; ++idcam) {
      cv::Mat mask;

      /*Retrieve mask*/
      calibConfig.getControlPointsMask(mask, idcam, idinput);

      /*Do the real extraction part*/
      std::shared_ptr<CvImage> img = rigInputImages[idcam][idinput];
      FAIL_RETURN(kpExtractor.extract(*img.get(), keypoints[idcam], descriptors[idcam], mask));
      FAIL_RETURN(progress.add(CalibrationProgress::kpDetect, "Detecting"));
      Logger::get(Logger::Verbose) << "Found " << keypoints[idcam].size() << " points in camera #" << idcam
                                   << std::endl;
    }

    /*Perform matching for all pairs*/
    for (videoreaderid_t idcam1 = 0; idcam1 < (videoreaderid_t)numCameras - 1; ++idcam1) {
      for (videoreaderid_t idcam2 = idcam1 + 1; idcam2 < (videoreaderid_t)numCameras; ++idcam2) {
        /*Raw blind matching*/
        Core::ControlPointList matched;
        std::pair<unsigned int, unsigned int> pair{(unsigned int)idcam1, (unsigned int)idcam2};
        FAIL_RETURN(kpMatcher.match(frameNumbers[idinput], pair, keypoints[idcam1], descriptors[idcam1],
                                    keypoints[idcam2], descriptors[idcam2], matched));

        if (calibConfig.isSavingDebugSnapshots()) {
          drawMatches(rigInputImages, int(idinput), idcam1, idcam2, keypoints[idcam1], keypoints[idcam2], matched,
                      0 /* step 0 in calibration process */, std::string("rough"));
        }

        FAIL_RETURN(progress.add(CalibrationProgress::kpMatch, "Matching"));
        Logger::get(Logger::Verbose) << "Found " << matched.size() << " rough matched points between camera #" << idcam1
                                     << " and camera #" << idcam2 << std::endl;

        /*Merging result*/
        matchedpoints_map[pair].insert(matchedpoints_map[pair].end(), matched.begin(), matched.end());
      }
    }
  }

  return Status::OK();
}

Status Calibration::filterControlPoints(const double sphereScale, double* const cost) {
  FAIL_RETURN(progress.add(CalibrationProgress::filter, "Filtering"));

  /*Clear the points structures*/
  globalmatches_map.clear();

  double globalCost = 0.;

  for (const auto& matchlist : matchedpoints_map) {
    const auto pair = matchlist.first;
    const videoreaderid_t idcam1 = pair.first;
    const videoreaderid_t idcam2 = pair.second;
    Core::ControlPointList filtered;

    filterFromPresets(filtered, matchlist.second, idcam1, idcam2, sphereScale);

    if (Logger::getLevel() >= Logger::Verbose || calibConfig.isSavingDebugSnapshots() || cost != nullptr) {
      /*Project points from presets*/
      projectFromCurrentSettings(filtered, idcam1, idcam2, sphereScale);

      if (cost != nullptr) {
        /*Accumulate reprojection score*/
        globalCost += getMeanReprojectionDistance(filtered) - FILTERED_POINTS_WEIGHT_IN_SCORE * filtered.size();
      }
    }

    if (calibConfig.isSavingDebugSnapshots()) {
      drawMatches(rigInputImages, -1, idcam1, idcam2, KPList(), KPList(), filtered,
                  1 /* step 1 in calibration process */, std::string("filtered"));
      drawReprojectionErrors(rigInputImages, -1, idcam1, idcam2, KPList(), KPList(), filtered,
                             2 /* step 2 in calibration process */, std::string("filteredreprojected"));
    }

    Logger::get(Logger::Verbose) << "Found " << filtered.size() << " filtered matched points between camera #" << idcam1
                                 << " and camera #" << idcam2 << std::endl;

    /*Merging result*/
    globalmatches_map[pair].insert(globalmatches_map[pair].end(), filtered.begin(), filtered.end());
  }

  /*Output reprojection statistics*/
  if (Logger::getLevel() >= Logger::Verbose) {
    for (auto& matchlist : globalmatches_map) {
      reportProjectionStats(matchlist.second, matchlist.first.first, matchlist.first.second,
                            "Reprojection errors in pixels from presets");
    }
  }

  /*Return global score if necessary*/
  if (cost != nullptr) {
    *cost = globalCost;
  }

  return Status::OK();
}

bool Calibration::analyzeKeypointsConnectivity(
    const std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList>& keypoints_map,
    const videoreaderid_t numCameras, std::stringstream* reportString,
    std::map<videoreaderid_t, std::set<videoreaderid_t>>* connectivityPtr,
    std::set<videoreaderid_t>* singleConnectedInputsPtr, std::set<videoreaderid_t>* nonConnectedInputsPtr) {
  /* Analyze and log the inputs connectivity */
  std::map<videoreaderid_t, std::set<videoreaderid_t>> connectivity;
  std::set<videoreaderid_t> singleConnectedInputs;
  std::set<videoreaderid_t> nonConnectedInputs;
  for (auto& matchlist : keypoints_map) {
    auto pair = matchlist.first;
    // need at least 3 matched points per pair to estimate relative rotations
    if (matchlist.second.size() < 3) {
      continue;
    }
    connectivity[pair.first].insert(pair.second);
    connectivity[pair.second].insert(pair.first);
  }
  for (videoreaderid_t camId = 0; camId < numCameras; ++camId) {
    std::stringstream message;
    switch (connectivity[camId].size()) {
      case 0:
        nonConnectedInputs.insert(camId);
        message << "Camera " << camId << " is not connected to any other camera. " << std::endl;
        Logger::get(Logger::Error) << message.str() << std::flush;
        if (reportString) {
          *reportString << message.str();
        }
        break;
      case 1:
        singleConnectedInputs.insert(camId);
        message << "Camera " << camId << " is connected to a single camera " << containerToString(connectivity[camId])
                << " - calibration may not be optimal. " << std::endl;
        Logger::get(Logger::Warning) << message.str() << std::flush;
        if (reportString) {
          *reportString << message.str();
        }
        break;
      default:
        message << "Camera " << camId << " is connected to " << connectivity[camId].size() << " cameras "
                << containerToString(connectivity[camId]) << std::endl;
        Logger::get(Logger::Verbose) << message.str() << std::flush;
        break;
    }
  }
  if (!singleConnectedInputs.empty()) {
    std::stringstream message;
    message << "Missing camera connections that may improve the calibration: "
            << containerToString(singleConnectedInputs) << ". " << std::endl;
    Logger::get(Logger::Warning) << message.str() << std::flush;
    if (reportString) {
      *reportString << message.str();
    }
  }
  bool fullyConnected = nonConnectedInputs.empty();
  // return found sets if necessary
  containerSwapIfPtr(connectivityPtr, connectivity);
  containerSwapIfPtr(singleConnectedInputsPtr, singleConnectedInputs);
  containerSwapIfPtr(nonConnectedInputsPtr, nonConnectedInputs);

  // return true if all inputs are connected
  return fullyConnected;
}

Status Calibration::estimateExtrinsics(double& cost) {
  FAIL_RETURN(progress.add(CalibrationProgress::initGeometry, "Finding geometry"));

  ControlPointFilter cpfilter(calibConfig.getDecimatingGridSize(), calibConfig.getAngleThreshold(),
                              calibConfig.getMinRatioInliers(), calibConfig.getMinSamplesForFit(),
                              calibConfig.getRatioOutliers(), calibConfig.getProbaDrawOutlierFree());
  RigGraph::EdgeList relativeRotations;
  cost = std::numeric_limits<double>::max();

  /*Clear the points structures*/
  extrinsicsmatches_map.clear();

  /*Filtering out outliers by rotation model fitting*/
  for (auto& matchlist : globalmatches_map) {
    Core::ControlPointList filteredControlPoints;
    unsigned int source = matchlist.first.first;
    unsigned int dest = matchlist.first.second;
    const std::shared_ptr<Camera> camera1 = cameras[source];
    const std::shared_ptr<Camera> camera2 = cameras[dest];
    double meanReprojectionDistance = 0.;

    Logger::get(Logger::Debug) << "CAM: " << source << " -> " << dest << "   before filtering" << std::endl;
    /*Reset the seed to have deshuffling produce steady results*/
    gen.seed(CALIBRATION_RANDOM_SEED);
    const bool success =
        cpfilter.filterFromExtrinsics(filteredControlPoints, camera1, camera2, matchlist.second,
                                      configmatches_map[{source, dest}], syntheticmatches_map[{source, dest}], gen);
    Logger::get(Logger::Debug) << "CAM: " << source << " -> " << dest << "   after filtering" << std::endl;

    if (!success) {
      Logger::get(Logger::Verbose) << "Could not find rotation between camera #" << source << " and camera #" << dest
                                   << std::endl;
      continue;
    }
    Logger::get(Logger::Debug) << "CAM: " << source << " -> " << dest << "   FILTERING OK" << std::endl;

    /*Update the reprojections and get the errors*/
    cpfilter.projectFromEstimatedRotation(filteredControlPoints, camera1, camera2);
    meanReprojectionDistance = getMeanReprojectionDistance(filteredControlPoints);

    if (Logger::getLevel() >= Logger::Verbose || calibConfig.isSavingDebugSnapshots()) {
      if (calibConfig.isSavingDebugSnapshots()) {
        drawMatches(rigInputImages, -1 /* all pictures */, source, dest, KPList(), KPList(), filteredControlPoints,
                    3 /* step 3 in calibration process */, std::string("extrinsics"));
        drawReprojectionErrors(rigInputImages, -1 /* all pictures */, source, dest, KPList(), KPList(),
                               filteredControlPoints, 4 /* step 4 in calibration process */,
                               std::string("extrinsicsreprojected"));
      }
    }

    if (Logger::getLevel() >= Logger::Verbose) {
      reportProjectionStats(filteredControlPoints, source, dest,
                            "Reprojection errors in pixels from estimated extrinsics");
    }

    /*Merging result*/
    extrinsicsmatches_map[matchlist.first].insert(extrinsicsmatches_map[matchlist.first].end(),
                                                  filteredControlPoints.begin(), filteredControlPoints.end());

    /*Use a combination of extrinsics rotation score and number of inliers, the smaller, the better*/
    double weight = meanReprojectionDistance - FILTERED_POINTS_WEIGHT_IN_SCORE * cpfilter.getConsensus();
    RigGraph::WeightedEdge wedge(weight, source, dest, cpfilter.getEstimatedRotation());
    relativeRotations.push_back(wedge);
    Logger::get(Logger::Verbose) << "Found " << filteredControlPoints.size()
                                 << " statistically filtered matched points between camera #" << matchlist.first.first
                                 << " and camera #" << matchlist.first.second << std::endl;
  }

  /* Analyze connectivity, do not care about the return value */
  std::stringstream report;
  analyzeKeypointsConnectivity(extrinsicsmatches_map, (videoreaderid_t)cameras.size(), &report);

  /* Create input graph using all control points generated so far */
  RigGraph graph(cameras.size(), relativeRotations);

  /* Fail if  graph is not connected */
  if (!graph.isConnected()) {
    return {Origin::CalibrationAlgorithm, ErrType::AlgorithmFailure,
            report.str() +
                "Not enough control points were found, inputs are not fully connected to each other. Please, rotate "
                "your rig and repeat the process."};
  }

  cost = RigBuilder::build(cameras, graph, 0);
  return Status::OK();
}

Status Calibration::optimize(Core::PanoDefinition& pano) {
  FAIL_RETURN(progress.add(CalibrationProgress::optim, "Optimizing"));
#ifndef CERESLIB_UNSUPPORTED
  CalibrationRefinement refiner;
  refiner.setupWithCameras(cameras);
  FAIL_RETURN(refiner.process(pano, extrinsicsmatches_map, calibConfig));
#endif

  if (Logger::getLevel() >= Logger::Verbose || calibConfig.isSavingDebugSnapshots()) {
    /* Get some statistics after refinement */
    for (auto& matchlist : extrinsicsmatches_map) {
      Core::ControlPointList filteredControlPoints = matchlist.second;
      unsigned int source = matchlist.first.first;
      unsigned int dest = matchlist.first.second;
      projectFromCurrentSettings(filteredControlPoints, source, dest, pano.getSphereScale());

      if (calibConfig.isSavingDebugSnapshots()) {
        drawReprojectionErrors(rigInputImages, -1 /* all pictures */, source, dest, KPList(), KPList(),
                               filteredControlPoints, 5 /* step 5 in calibration process */,
                               std::string("extrinsicsrefinedreprojected"));
      }

      if (Logger::getLevel() >= Logger::Verbose) {
        reportProjectionStats(filteredControlPoints, source, dest,
                              "Reprojection errors in pixels from refined parameters, on extrinsics-filtered points");
      }
    }

#if REPORT_STATS_ON_FILTERED_POINTS
    for (auto& matchlist : globalmatches_map) {
      ControlPointList filteredControlPoints = matchlist.second;

      unsigned int source = matchlist.first.first;
      unsigned int dest = matchlist.first.second;

      projectFromPresets(filteredControlPoints, source, dest, pano.getSphereScale());

      if (calibConfig.isSavingDebugSnapshots()) {
        drawReprojectionErrors(rigInputImages, -1 /* all pictures */, source, dest, KPList(), KPList(),
                               filteredControlPoints, 6 /* step 6 in calibration process */,
                               std::string("filteredrefinedreprojected"));
      }

      if (Logger::getLevel() >= Logger::Verbose) {
        reportProjectionStats(filteredControlPoints, source, dest,
                              "Reprojection errors in pixels from refined parameters, on filtered points");
      }
    }
#endif
  }

  return Status::OK();
}

Status Calibration::calculateFOV(Core::PanoDefinition& pano) {
  /*FOV is not defined, loop over predetermined list of values (brute force every angle between FOV_ITERATE_START and
   * FOV_ITERATE_END degrees)*/
  VideoStitch::Core::RigCameraDefinition cam;
  calibConfig.getRigPreset()->getRigCameraDefinition(cam, 0);
  const VideoStitch::Core::InputDefinition::Format lensType = cam.getCamera()->getType();
  const std::vector<double> fovValues = getFOVValues();
  const auto videoInputs = pano.getVideoInputs();
  const VideoStitch::Core::InputDefinition& firstVideoInput = videoInputs[0];
  double currentBestCalibrationCost = std::numeric_limits<double>::max();
  std::size_t bestIdFov = 0;

  for (std::size_t indexFov = 0; indexFov < fovValues.size(); ++indexFov) {
    CalibrationConfig currentConfig(calibConfig);
    /* disable saving debug snapshots while iterating on FOV values */
    currentConfig.setIsSavingDebugSnapshots(false);
    /* disable generating artificial keypoints */
    currentConfig.setIsGeneratingSyntheticKeypoints(false);
    /* check that we are not deshuffling the inputs */
    if (currentConfig.isInDeshuffleMode()) {
      return {Origin::CalibrationAlgorithm, ErrType::InvalidConfiguration,
              "Deshuffling and automatic FOV determination are incompatible"};
    }
    currentConfig.setRigDefinition(
        std::shared_ptr<VideoStitch::Core::RigDefinition>(VideoStitch::Core::RigDefinition::createBasicUnknownRig(
            "default", lensType, pano.numVideoInputs(), firstVideoInput.getWidth(), firstVideoInput.getHeight(),
            firstVideoInput.getCroppedWidth(), firstVideoInput.getCroppedHeight(), fovValues[indexFov])));
    std::unique_ptr<VideoStitch::Core::PanoDefinition> currentPanoDef(pano.clone());
    FAIL_RETURN(progress.add(CalibrationProgress::fovIterate / static_cast<double>(fovValues.size()),
                             "Optimizing the field of view (FOV)"));
    updateCalibrationConfig(currentConfig);
    if (!processGivenMatchedControlPoints(*currentPanoDef, false /* do not perform full optimization */).ok()) {
      continue;
    }
    if (currentPanoDef->getCalibrationCost() <= currentBestCalibrationCost) {
      currentBestCalibrationCost = currentPanoDef->getCalibrationCost();
      bestIdFov = indexFov;
    }
  }
  CalibrationConfig currentConfig(calibConfig);
  Logger::get(Logger::Debug) << "Computing BEST calibration with: " << fovValues[bestIdFov] << std::endl;
  currentConfig.setRigDefinition(
      std::shared_ptr<VideoStitch::Core::RigDefinition>(VideoStitch::Core::RigDefinition::createBasicUnknownRig(
          "default", lensType, pano.numVideoInputs(), firstVideoInput.getWidth(), firstVideoInput.getHeight(),
          firstVideoInput.getCroppedWidth(), firstVideoInput.getCroppedHeight(), fovValues[bestIdFov], &pano)));
  updateCalibrationConfig(currentConfig);
  initialHFOV = fovValues[bestIdFov];
  return Status::OK();
}

void Calibration::filterFromPresets(Core::ControlPointList& filtered, const Core::ControlPointList& input,
                                    videoreaderid_t idcam1, videoreaderid_t idcam2, const double sphereScale) {
  const std::shared_ptr<Camera> camera1 = cameras[idcam1];
  const std::shared_ptr<Camera> camera2 = cameras[idcam2];

  std::shared_ptr<Core::RigDefinition> rig = calibConfig.getRigPreset();
  Core::RigCameraDefinition rigcamdef1, rigcamdef2;
  std::shared_ptr<Core::CameraDefinition> rigcam1, rigcam2;
  rig->getRigCameraDefinition(rigcamdef1, idcam1);
  rig->getRigCameraDefinition(rigcamdef2, idcam2);
  rigcam1 = rigcamdef1.getCamera();
  rigcam2 = rigcamdef2.getCamera();
  Eigen::Vector3d meanpt3d;
  Eigen::Matrix3d cov3d;
  Eigen::Matrix2d cov2d;

  for (Core::ControlPointList::const_iterator it = input.begin(); it != input.end(); ++it) {
    Eigen::Vector2d impt1, impt2_mean, impt2_real;
    impt1(0) = it->x0;
    impt1(1) = it->y0;

    impt2_real(0) = it->x1;
    impt2_real(1) = it->y1;

    bool res = camera1->getLiftCovariance(
        meanpt3d, cov3d, rigcam1->getFu().variance, rigcam1->getFv().variance, rigcam1->getCu().variance,
        rigcam1->getCv().variance, rigcam1->getDistortionA().variance, rigcam1->getDistortionB().variance,
        rigcam1->getDistortionC().variance, rigcamdef1.getTranslationX().variance,
        rigcamdef1.getTranslationY().variance, rigcamdef1.getTranslationZ().variance, impt1, sphereScale);
    if (!res) {
      continue;
    }

    res = camera2->getProjectionCovariance(
        impt2_mean, cov2d, rigcam2->getFu().variance, rigcam2->getFv().variance, rigcam2->getCu().variance,
        rigcam2->getCv().variance, rigcam2->getDistortionA().variance, rigcam2->getDistortionB().variance,
        rigcam2->getDistortionC().variance, rigcamdef2.getTranslationX().variance,
        rigcamdef2.getTranslationY().variance, rigcamdef2.getTranslationZ().variance, meanpt3d, cov3d);
    if (!res) {
      continue;
    }

    /*Check Mahalanobis distance*/
    Eigen::Vector2d diff = impt2_real - impt2_mean;
    Eigen::Matrix2d invcov = cov2d.inverse();

    // using the 3.sigmas rule for Normal distributions: the Mahalanobis distance is a squared norm, use 3.0^2 as the
    // threshold
    if ((diff.transpose() * invcov * diff) > 9.0) {
      continue;
    }

    filtered.push_back(*it);
  }
}

void Calibration::fillPanoWithControlPoints(Core::PanoDefinition& pano) {
  // aggregate the list of control points that passed the extrinsics stage
  Core::ControlPointList full_extrinsics_list;
  for (auto& matchlist : extrinsicsmatches_map) {
    full_extrinsics_list.insert(full_extrinsics_list.end(), matchlist.second.begin(), matchlist.second.end());
  }

  // remove synthetic control points from list, if any
  if (calibConfig.isGeneratingSyntheticKeypoints()) {
    full_extrinsics_list.remove_if([](const Core::ControlPoint& cp) -> bool { return cp.artificial; });
  }

  pano.setCalibrationControlPointList(full_extrinsics_list);
  Logger::get(Logger::Info) << "Calibration: returning " << full_extrinsics_list.size() << " control points"
                            << std::endl;

  if (Logger::getLevel() >= Logger::Verbose) {
    reportControlPointsStats(full_extrinsics_list);
  }
}

void Calibration::projectFromCurrentSettings(Core::ControlPointList& input, videoreaderid_t idcam1,
                                             videoreaderid_t idcam2, const double sphereScale) {
  const std::shared_ptr<Camera> camera1 = cameras[idcam1];
  const std::shared_ptr<Camera> camera2 = cameras[idcam2];
  Eigen::Vector3d meanpt3d;
  Eigen::Vector2d meanpt2d;

  for (auto& it : input) {
    it.rx0 = it.ry0 = it.rx1 = it.ry1 = 0.;
    Eigen::Vector2d impt1(it.x0, it.y0), impt2(it.x1, it.y1);
    // project impt1 onto second camera
    bool res = camera1->lift(meanpt3d, impt1, sphereScale);
    if (!res) {
      continue;
    }
    res = camera2->project(meanpt2d, meanpt3d);
    if (!res) {
      continue;
    }

    // store the projection result
    it.rx0 = meanpt2d(0);
    it.ry0 = meanpt2d(1);
    // project impt2 onto first camera
    res = camera2->lift(meanpt3d, impt2, sphereScale);
    if (!res) {
      continue;
    }
    res = camera1->project(meanpt2d, meanpt3d);
    if (!res) {
      continue;
    }
    // store the projection result
    it.rx1 = meanpt2d(0);
    it.ry1 = meanpt2d(1);
  }
}

Status Calibration::applyPresetsGeometry(Core::PanoDefinition& pano) {
  FAIL_RETURN(fillPano(pano, cameras));
  return Status::OK();
}

}  // namespace Calibration
}  // namespace VideoStitch

#endif  // __clang_analyzer__
