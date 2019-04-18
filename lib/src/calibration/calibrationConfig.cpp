// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationConfig.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/rigDef.hpp"
#include "libvideostitch/rigCameraDef.hpp"
#include "libvideostitch/cameraDef.hpp"

namespace VideoStitch {
namespace Calibration {

#define DETECTION_OCTAVES 2
#define DETECTION_LEVELS 2
#define DETECTION_THRESHOLD 0.001
#define MATCHING_DISTANCE 0.5         // 0.3
#define FILTER_CELLFACTOR 0.02        // 0.04
#define FILTER_ANGLE_THRESHOLD 5.0    // 2.0
#define FILTER_MIN_RATIO_INLIERS 0.1  // 0.7
#define FILTER_MIN_SAMPLES 3
#define FILTER_RATIO_OUTLIERS 0.5
#define FILTER_PROBA_INLIERS 0.99

#define ORAH_4I_PRESETS_FOCAL_STDDEV_PERCENTAGE 0.1   // +/-3*0.1% using the 3-sigma rule
#define ORAH_4I_PRESETS_CENTER_STDDEV_PIXELS 3.       // +/-3*3 pixels using the 3-sigma rule
#define ORAH_4I_PRESETS_DISTORT_STDDEV_PERCENTAGE 0.  // keep distortion parameters constant
#define ORAH_4I_PRESETS_ANGLE_STDDEV 1                // +/-3*1 degrees using the 3-sigma rule
#define ORAH_4I_PRESETS_TRANSLATION_STDDEV 0.         // keep translations constant

#define SYNTHETIC_KEYPOINTS_GRID_WIDTH 5.
#define SYNTHETIC_KEYPOINTS_GRID_HEIGHT 5.

CalibrationConfig::CalibrationConfig(const Ptv::Value* config)
    : isConfigValid(true),
      applyPresetsOnly(false),
      deshuffleMode(false),
      deshuffleModeOnly(false),
      deshuffleModePreserveReadersOrder(false),
      automaticFovIterate(false),
      initialHFovValue(0.0),
      singleFocalForAllLenses(false),
      improveMode(false),
      dumpDebugSnapshots(false),
      useSyntheticKeypoints(false),
      syntheticKeypointsGridWidth(SYNTHETIC_KEYPOINTS_GRID_WIDTH),
      syntheticKeypointsGridHeight(SYNTHETIC_KEYPOINTS_GRID_HEIGHT),
      extractor("AKAZE"),
      matcher("HAMMING"),
      octaves(DETECTION_OCTAVES),
      sublevels(DETECTION_LEVELS),
      threshold(DETECTION_THRESHOLD),
      nndr_ratio(MATCHING_DISTANCE),
      angle_threshold(FILTER_ANGLE_THRESHOLD),
      min_ratio_inliers(FILTER_MIN_RATIO_INLIERS),
      min_samples_for_fit(FILTER_MIN_SAMPLES),
      ratio_outliers(FILTER_RATIO_OUTLIERS),
      proba_draw_outlier_free(FILTER_PROBA_INLIERS),
      decimating_grid_size(FILTER_CELLFACTOR) {
  if (!config) {
    isConfigValid = false;
    return;
  }

#define LOG_WRONGTYPE(call)                                                      \
  if (call == Parse::PopulateResult_WrongType) {                                 \
    Logger::get(Logger::Error) << #call << " returned wrong type " << std::endl; \
  }

  LOG_WRONGTYPE(Parse::populateBool("CalibrationConfig", *config, "apply_presets_only", applyPresetsOnly, false))
  LOG_WRONGTYPE(Parse::populateBool("CalibrationConfig", *config, "deshuffle_mode", deshuffleMode, false))
  LOG_WRONGTYPE(Parse::populateBool("CalibrationConfig", *config, "deshuffle_mode_only", deshuffleModeOnly, false))
  LOG_WRONGTYPE(Parse::populateBool("CalibrationConfig", *config, "deshuffle_mode_preserve_readers_order",
                                    deshuffleModePreserveReadersOrder, false))
  LOG_WRONGTYPE(Parse::populateBool("CalibrationConfig", *config, "auto_iterate_fov", automaticFovIterate, false))
  LOG_WRONGTYPE(Parse::populateDouble("CalibrationConfig", *config, "initial_hfov", initialHFovValue, false))
  LOG_WRONGTYPE(Parse::populateBool("CalibrationConfig", *config, "single_focal", singleFocalForAllLenses, false))
  LOG_WRONGTYPE(Parse::populateBool("CalibrationConfig", *config, "improve_mode", improveMode, false))
  LOG_WRONGTYPE(
      Parse::populateBool("CalibrationConfig", *config, "dump_calibration_snapshots", dumpDebugSnapshots, false))
  LOG_WRONGTYPE(
      Parse::populateBool("CalibrationConfig", *config, "use_synthetic_keypoints", useSyntheticKeypoints, false))
  LOG_WRONGTYPE(Parse::populateDouble("CalibrationConfig", *config, "synthetic_keypoints_grid_width",
                                      syntheticKeypointsGridWidth, false))
  LOG_WRONGTYPE(Parse::populateDouble("CalibrationConfig", *config, "synthetic_keypoints_grid_height",
                                      syntheticKeypointsGridHeight, false))

  /**
   * Basic checks on general calibration options
   */
  if (useSyntheticKeypoints && (syntheticKeypointsGridWidth <= 1 || syntheticKeypointsGridHeight <= 1)) {
    Logger::get(Logger::Error) << "Invalid synthetic keypoints grid width or height, must be > 1" << std::endl;
    isConfigValid = false;
  }

  /*DeshuffleModeOnly implies deshuffleMode*/
  deshuffleMode |= deshuffleModeOnly;

  /**
   * Parse control point extractor
   */
  const Ptv::Value* cp_extractor = config->has("cp_extractor");
  if (cp_extractor) {
    LOG_WRONGTYPE(Parse::populateString("cp_extractor", *cp_extractor, "extractor", extractor, false))
    LOG_WRONGTYPE(Parse::populateString("cp_extractor", *cp_extractor, "matcher_norm", matcher, false))
    LOG_WRONGTYPE(Parse::populateInt("cp_extractor", *cp_extractor, "octaves", octaves, false))
    LOG_WRONGTYPE(Parse::populateInt("cp_extractor", *cp_extractor, "sublevels", sublevels, false))
    LOG_WRONGTYPE(Parse::populateDouble("cp_extractor", *cp_extractor, "threshold", threshold, false))
    LOG_WRONGTYPE(Parse::populateDouble("cp_extractor", *cp_extractor, "nndr_ratio", nndr_ratio, false))

    /*Some basic checks, to be extended when more extractors and descriptors are supported*/
    if (!(extractor == "AKAZE") || !(matcher == "HAMMING")) {
      Logger::get(Logger::Error) << "Invalid extrator and/or matcher types in cp_extractor" << std::endl;
      isConfigValid = false;
    }
  }

  /**
   * Parse control point filter
   */
  const Ptv::Value* cp_filter = config->has("cp_filter");
  if (cp_extractor) {
    LOG_WRONGTYPE(Parse::populateDouble("cp_filter", *cp_filter, "angle_threshold", angle_threshold, false))
    LOG_WRONGTYPE(Parse::populateDouble("cp_filter", *cp_filter, "min_ratio_inliers", min_ratio_inliers, false))
    LOG_WRONGTYPE(Parse::populateInt("cp_filter", *cp_filter, "min_samples_for_fit", min_samples_for_fit, false))
    LOG_WRONGTYPE(Parse::populateDouble("cp_filter", *cp_filter, "ratio_outliers", ratio_outliers, false))
    LOG_WRONGTYPE(Parse::populateDouble("cp_filter", *cp_filter, "proba_draw_outlier_free_samples",
                                        proba_draw_outlier_free, false))
    LOG_WRONGTYPE(Parse::populateDouble("cp_filter", *cp_filter, "decimating_grid_size", decimating_grid_size, false))

    /*Some basic checks, to be extended when more extractors and descriptors are supported*/
    if (angle_threshold <= 0. || min_ratio_inliers <= 0. || min_samples_for_fit <= 0 || ratio_outliers <= 0. ||
        proba_draw_outlier_free <= 0. || proba_draw_outlier_free >= 1. || decimating_grid_size <= 0. ||
        decimating_grid_size >= 1.) {
      Logger::get(Logger::Error) << "Invalid parameters in cp_filter" << std::endl;
      isConfigValid = false;
    }
  }

  /**
   * Parse list of video frames
   */
  const Ptv::Value* val_list_frames = config->has("list_frames");
  if (val_list_frames && val_list_frames->getType() == Ptv::Value::LIST) {
    std::vector<Ptv::Value*> listFramesPTV = val_list_frames->asList();
    for (auto& f : listFramesPTV) {
      frames.push_back((unsigned int)f->asInt());
    }
    /*Make sure list of frames is sorted in increasing order, to speed-up seek operations*/
    std::sort(frames.begin(), frames.end());
  }
  if (frames.empty() && !applyPresetsOnly) {
    Logger::get(Logger::Error) << "Invalid or missing frames list in calibration configuration" << std::endl;
    isConfigValid = false;
  }

  /**
   *List over cameras presets
   */
  std::map<std::string, std::shared_ptr<Core::CameraDefinition> > cameras_map;
  const Ptv::Value* val_list_cameras = config->has("cameras");
  if (val_list_cameras && val_list_cameras->getType() == Ptv::Value::LIST) {
    std::vector<Ptv::Value*> list_cameras = val_list_cameras->asList();

    /*Loop over list of camera presets*/
    for (auto& f : list_cameras) {
      std::shared_ptr<Core::CameraDefinition> cam(Core::CameraDefinition::create(*f));
      if (cam.get()) {
        cameras_map[cam->getName()] = cam;
      } else {
        isConfigValid = false;
      }
    }
  }
  if (cameras_map.empty()) {
    Logger::get(Logger::Error) << "Invalid or missing cameras definition in calibration configuration" << std::endl;
    isConfigValid = false;
  }

  /**
   * Load rig preset
   */
  const Ptv::Value* val_rig = config->has("rig");
  if (val_rig && val_rig->getType() == Ptv::Value::OBJECT) {
    rig.reset(Core::RigDefinition::create(cameras_map, *val_rig));
    if (rig.get() == nullptr) {
      isConfigValid = false;
    }
  }
  if (!rig) {
    Logger::get(Logger::Error) << "Invalid or missing rig definition in calibration configuration" << std::endl;
    isConfigValid = false;
  }

  /**
   * If the rig corresponds to an Orah 4i camera, override the presets from factory
   */
  if (rig && rig->getName() == "Orah 4i") {
    Logger::get(Logger::Info) << "Orah 4i presets detected" << std::endl;

    rig->overridePresetsStandardDeviations(
        ORAH_4I_PRESETS_FOCAL_STDDEV_PERCENTAGE,
        ORAH_4I_PRESETS_CENTER_STDDEV_PIXELS * 100. / 1920 /* converting to a percentage of Orah 4i picture width */,
        ORAH_4I_PRESETS_DISTORT_STDDEV_PERCENTAGE, ORAH_4I_PRESETS_ANGLE_STDDEV, ORAH_4I_PRESETS_ANGLE_STDDEV,
        ORAH_4I_PRESETS_ANGLE_STDDEV, ORAH_4I_PRESETS_TRANSLATION_STDDEV, ORAH_4I_PRESETS_TRANSLATION_STDDEV,
        ORAH_4I_PRESETS_TRANSLATION_STDDEV);
  }

  /**
   * A mask define for a given camera and frame, a per pixel value.
   * This value, if 0 will deny extraction of features at this particular coordinate.
   */
  const Ptv::Value* val_list_masks = config->has("masks");
  if (val_list_masks && val_list_masks->getType() == Ptv::Value::LIST) {
    std::vector<Ptv::Value*> list_masks = val_list_masks->asList();

    /*Loop over list of encoded masks*/
    for (auto& f : list_masks) {
      if (f->getType() == Ptv::Value::OBJECT) {
        const Ptv::Value* val_frameid = f->has("frame_id");
        const Ptv::Value* val_camid = f->has("camera_id");
        const Ptv::Value* val_content = f->has("content");

        /*If object is fully valid*/
        if (val_frameid && val_camid && val_content) {
          if (val_frameid->getType() == Ptv::Value::INT && val_camid->getType() == Ptv::Value::INT &&
              val_content->getType() == Ptv::Value::STRING) {
            std::string str;
            size_t camid = (size_t)val_camid->asInt();
            int frameid = (int)val_camid->asInt();

            if (frameid < 0) {
              continue;
            }
            if (camid > rig->getRigCameraDefinitionCount()) {
              continue;
            }

            str = val_content->asString();

            Core::RigCameraDefinition rigcam;
            if (!rig->getRigCameraDefinition(rigcam, camid)) {
              continue;
            }

            size_t width = rigcam.getCamera()->getWidth();
            size_t height = rigcam.getCamera()->getHeight();
            if (str.length() > width * height) {
              Logger::get(Logger::Error) << "A Control point mask is defined but its size is incorrect" << std::endl;
              continue;
            }

            cv::Mat mask = cv::Mat((int)height, (int)width, CV_8UC1);
            int pos = 0;
            for (int i = 0; i < (int)height; i++) {
              for (int j = 0; j < (int)width; j++) {
                if (str[pos] == '0') {
                  mask.at<unsigned char>(i, j) = 0;
                } else {
                  mask.at<unsigned char>(i, j) = 255;
                }
                pos++;
              }
            }

            /*We do not test input size as we do not know it ?*/
            if (!str.empty()) {
              std::pair<size_t, size_t> key;
              key.first = camid;
              key.second = frameid;
              masksmap[key] = mask;
            }
          }
        }
      }
    }
  }

  /**
   * Load calibration control points, if any
   */
  if (improveMode) {
    Potential<Core::ControlPointListDefinition> cpListDef = Core::ControlPointListDefinition::create(*config);
    if (cpListDef.ok()) {
      cpList = cpListDef->getCalibrationControlPointList();
      Logger::get(Logger::Info) << "Calibration: reusing " << cpList.size()
                                << " control point(s) from former calibration(s)" << std::endl;
    }
  }
  if (cpList.empty() && !applyPresetsOnly) {
    Logger::get(Logger::Info) << "Calibration: starting calibration from scratch" << std::endl;
    improveMode = false;
  }

#undef LOG_WRONGTYPE
}

CalibrationConfig::CalibrationConfig(const CalibrationConfig& other)
    : isConfigValid(other.isConfigValid),
      applyPresetsOnly(other.applyPresetsOnly),
      deshuffleMode(other.deshuffleMode),
      deshuffleModeOnly(other.deshuffleModeOnly),
      deshuffleModePreserveReadersOrder(other.deshuffleModePreserveReadersOrder),
      automaticFovIterate(other.automaticFovIterate),
      initialHFovValue(other.initialHFovValue),
      singleFocalForAllLenses(other.singleFocalForAllLenses),
      improveMode(other.improveMode),
      dumpDebugSnapshots(other.dumpDebugSnapshots),
      useSyntheticKeypoints(other.useSyntheticKeypoints),
      syntheticKeypointsGridWidth(other.syntheticKeypointsGridWidth),
      syntheticKeypointsGridHeight(other.syntheticKeypointsGridHeight),
      frames(other.frames),
      masksmap(other.masksmap),
      rig(other.rig),
      cpList(other.cpList),
      extractor(other.extractor),
      matcher(other.matcher),
      octaves(other.octaves),
      sublevels(other.sublevels),
      threshold(other.threshold),
      nndr_ratio(other.nndr_ratio),
      angle_threshold(other.angle_threshold),
      min_ratio_inliers(other.min_ratio_inliers),
      min_samples_for_fit(other.min_samples_for_fit),
      ratio_outliers(other.ratio_outliers),
      proba_draw_outlier_free(other.proba_draw_outlier_free),
      decimating_grid_size(other.decimating_grid_size) {}

}  // namespace Calibration
}  // namespace VideoStitch
