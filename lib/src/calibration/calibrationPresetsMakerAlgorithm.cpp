// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationPresetsMakerAlgorithm.hpp"

#include "util/registeredAlgo.hpp"

#include "libvideostitch/rigDef.hpp"
#include "libvideostitch/rigCameraDef.hpp"
#include "libvideostitch/cameraDef.hpp"

#include <iostream>

namespace VideoStitch {
namespace CalibrationPresetsMaker {

namespace {
Util::RegisteredAlgo<CalibrationPresetsMakerAlgorithm> registered("calibration_presets_maker");
}

const char* CalibrationPresetsMakerAlgorithm::docString =
    "An algorithm that takes an input PanoDefinition and creates calibration presets for it.\n";

CalibrationPresetsMakerAlgorithm::CalibrationPresetsMakerAlgorithm(const Ptv::Value* config)
    : calibrationPresetsMakerConfig(config) {}

CalibrationPresetsMakerAlgorithm::~CalibrationPresetsMakerAlgorithm() {}

Potential<Ptv::Value> CalibrationPresetsMakerAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter*,
                                                              Util::OpaquePtr**) const {
  if (pano == nullptr) {
    return {Origin::CalibrationAlgorithm, ErrType::InvalidConfiguration, "PanoDefinition is empty"};
  }

  std::unique_ptr<VideoStitch::Core::RigDefinition> rigdef(
      VideoStitch::Core::RigDefinition::createFromPanoDefinitionTemplate(
          calibrationPresetsMakerConfig.getPresetsName(), calibrationPresetsMakerConfig.getFocalStdDevValuePercentage(),
          calibrationPresetsMakerConfig.getCenterStdDevWidthPercentage(),
          calibrationPresetsMakerConfig.getDistortStdDevValuePercentage(), calibrationPresetsMakerConfig.getYawStdDev(),
          calibrationPresetsMakerConfig.getPitchStdDev(), calibrationPresetsMakerConfig.getRollStdDev(),
          calibrationPresetsMakerConfig.getTranslationXStdDev(), calibrationPresetsMakerConfig.getTranslationYStdDev(),
          calibrationPresetsMakerConfig.getTranslationZStdDev(), *pano, true));

  if (rigdef == nullptr) {
    return {Origin::CalibrationAlgorithm, ErrType::AlgorithmFailure, "could not create rig definition from template"};
  }

  /*Create the result*/
  Potential<Ptv::Value> res(Ptv::Value::emptyObject());

  res->push("rig", rigdef->serialize());

  VideoStitch::Ptv::Value* listCameras = VideoStitch::Ptv::Value::emptyObject();
  for (auto it : rigdef->getRigCameraDefinitionMap()) {
    listCameras->asList().push_back(it.second->serialize());
  }
  res->push("cameras", listCameras);

  /*Add rig presets to the PanoDefinition*/
  pano->setCalibrationRigPresets(rigdef.release());

  /*Remove control point list*/
  pano->setCalibrationControlPointList(Core::ControlPointList());

  return res;
}
}  // namespace CalibrationPresetsMaker
}  // namespace VideoStitch
