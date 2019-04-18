// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "epipolarCurvesConfig.hpp"

#include "libvideostitch/logging.hpp"

#include <algorithm>

#define DEFAULT_AUTO_POINT_MATCHING_VALUE true
#define DEFAULT_DECIMATION_CELL_FACTOR_VALUE 0.04
#define DEFAULT_SPHERICAL_GRID_RADIUS 0.
#define DEFAULT_IMAGE_MAX_OUTPUT_DEPTH 2.55

namespace VideoStitch {
namespace EpipolarCurves {

EpipolarCurvesConfig::EpipolarCurvesConfig(const Ptv::Value* config)
    : isConfigValid(false),
      autoPointMatching(DEFAULT_AUTO_POINT_MATCHING_VALUE),
      decimationCellFactor(DEFAULT_DECIMATION_CELL_FACTOR_VALUE),
      sphericalGridRadius(DEFAULT_SPHERICAL_GRID_RADIUS),
      imageMaxOutputDepth(DEFAULT_IMAGE_MAX_OUTPUT_DEPTH) {
  if (!config) {
    isConfigValid = false;
    return;
  }

#define RETURN_ON_WRONGTYPE(call)                \
  if (call == Parse::PopulateResult_WrongType) { \
    return;                                      \
  }
#define RETURN_ON_FAILURE(call)           \
  if (call != Parse::PopulateResult_Ok) { \
    return;                               \
  }

  RETURN_ON_WRONGTYPE(Parse::populateBool("config", *config, "auto_point_matching", autoPointMatching, false));
  RETURN_ON_WRONGTYPE(Parse::populateDouble("config", *config, "decimating_grid_size", decimationCellFactor, false));
  RETURN_ON_WRONGTYPE(Parse::populateDouble("config", *config, "spherical_grid_radius", sphericalGridRadius, false));
  RETURN_ON_WRONGTYPE(Parse::populateDouble("config", *config, "image_max_output_depth", imageMaxOutputDepth, false));

  /**
   * Parse list of video frames
   */
  const Ptv::Value* val_list_frames = config->has("list_frames");
  if (val_list_frames && val_list_frames->getType() == Ptv::Value::LIST) {
    std::vector<Ptv::Value*> listFramesPTV = val_list_frames->asList();
    for (auto& f : listFramesPTV) {
      frames.push_back((frameid_t)f->asInt());
    }
    /*Make sure list of frames is sorted in increasing order, to speed-up seek operations*/
    std::sort(frames.begin(), frames.end());
  }

  /*Default value if no frame list was given*/
  if (frames.empty()) {
    frames.push_back(0);
  }

  /**
   * Parse list of single points
   */
  const Ptv::Value* val_single_points = config->has("single_points");
  if (val_single_points && val_single_points->getType() == Ptv::Value::LIST) {
    std::vector<Ptv::Value*> points_list = val_single_points->asList();
    for (auto& c : points_list) {
      videoreaderid_t index;
      double x, y;

      RETURN_ON_FAILURE(Parse::populateInt("single_points", *c, "input_index", index, true));
      RETURN_ON_FAILURE(Parse::populateDouble("single_points", *c, "x", x, true));
      RETURN_ON_FAILURE(Parse::populateDouble("single_points", *c, "y", y, true));

      singlePointsMap[index].push_back(Core::TopLeftCoords2(static_cast<float>(x), static_cast<float>(y)));
    }
  }

  /**
   * Parse list of matched points
   */
  const Ptv::Value* val_pair_points = config->has("matched_points");
  if (val_pair_points && val_pair_points->getType() == Ptv::Value::LIST) {
    std::vector<Ptv::Value*> pair_points_list = val_pair_points->asList();
    for (auto& c : pair_points_list) {
      Core::ControlPoint cp;

      RETURN_ON_FAILURE(Parse::populateInt("matched_points", *c, "input_index0", cp.index0, true));
      RETURN_ON_FAILURE(Parse::populateDouble("matched_points", *c, "x0", cp.x0, true));
      RETURN_ON_FAILURE(Parse::populateDouble("matched_points", *c, "y0", cp.y0, true));
      RETURN_ON_FAILURE(Parse::populateInt("matched_points", *c, "input_index1", cp.index1, true));
      RETURN_ON_FAILURE(Parse::populateDouble("matched_points", *c, "x1", cp.x1, true));
      RETURN_ON_FAILURE(Parse::populateDouble("matched_points", *c, "y1", cp.y1, true));

      matchedPointsMap[{cp.index0, cp.index1}].push_back(cp);

      /* Add the points to the singlePointsMap too */
      singlePointsMap[cp.index0].push_back(Core::TopLeftCoords2(static_cast<float>(cp.x0), static_cast<float>(cp.y0)));
      singlePointsMap[cp.index1].push_back(Core::TopLeftCoords2(static_cast<float>(cp.x1), static_cast<float>(cp.y1)));
    }
  }

  isConfigValid = true;

#undef RETURN_ON_WRONGTYPE
#undef RETURN_ON_FAILURE
}

EpipolarCurvesConfig::EpipolarCurvesConfig(const EpipolarCurvesConfig& other)
    : isConfigValid(other.isConfigValid),
      autoPointMatching(other.autoPointMatching),
      decimationCellFactor(other.decimationCellFactor),
      sphericalGridRadius(other.sphericalGridRadius),
      imageMaxOutputDepth(other.imageMaxOutputDepth),
      frames(other.frames),
      singlePointsMap(other.singlePointsMap),
      matchedPointsMap(other.matchedPointsMap) {}

}  // namespace EpipolarCurves
}  // namespace VideoStitch
