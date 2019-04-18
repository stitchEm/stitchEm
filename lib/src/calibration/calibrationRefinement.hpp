// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/config.hpp"
#include "libvideostitch/rigDef.hpp"
#include "libvideostitch/controlPointListDef.hpp"

#include "calibrationConfig.hpp"

#include <memory>
#include <vector>

namespace VideoStitch {
namespace Calibration {

class Camera;

class VS_EXPORT CalibrationRefinement {
 public:
  CalibrationRefinement();
  virtual ~CalibrationRefinement();

  void setupWithCameras(const std::vector<std::shared_ptr<Camera> >& cameras);

  /**
  @brief Refines calibration parameters from the given control points
  @param pano the input output parameters for the panorama
  @param filteredCPMap map of ControlPoint lists, mapped by camera input pairs
  @param calibrationConfig calibration configuration object
  @return true on success, false on failure
  */
  Status process(Core::PanoDefinition& pano,
                 const std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList>& filteredCPMap,
                 const CalibrationConfig& calibrationConfig);

  std::shared_ptr<Camera> getCamera(size_t index);

 private:
  std::vector<std::shared_ptr<Camera> > cameras;
};

}  // namespace Calibration
}  // namespace VideoStitch
