// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationOnlineAlgorithm.hpp"

#include "calibration.hpp"
#include "cvImage.hpp"
#include "calibrationProgress.hpp"

#include "core/controllerInputFrames.hpp"
#include "util/registeredAlgo.hpp"
#include "cuda/memory.hpp"

#include "gpu/surface.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/rigDef.hpp"

#include <memory>
#include <functional>

namespace VideoStitch {
namespace Calibration {

namespace {
Util::RegisteredAlgo<CalibrationOnlineAlgorithm, true> registeredOnline("calibration");
}

const char* CalibrationOnlineAlgorithm::docString =
    "An algorithm that calibrates a panoramic multi-camera system using overlap zones between images\n";

CalibrationOnlineAlgorithm::CalibrationOnlineAlgorithm(const Ptv::Value* config) : CalibrationAlgorithmBase(config) {}

CalibrationOnlineAlgorithm::~CalibrationOnlineAlgorithm() {}

Status CalibrationOnlineAlgorithm::retrieveImages(
    RigCvImages& rig, const Core::PanoDefinition& pano,
    const std::vector<std::pair<videoreaderid_t, GPU::Surface&>>& frames) const {
  /*Create rig of n list*/
  rig.clear();
  rig.resize(pano.numVideoInputs());

  /* With non-video inputs, assume we can have gaps between camid and inputid */
  auto videoInputs = pano.getVideoInputs();
  for (auto it = frames.begin(); it != frames.end(); ++it) {
    int camid = it->first;

    if (camid >= (int)pano.numInputs() || camid >= (int)videoInputs.size()) {
      return {Origin::CalibrationAlgorithm, ErrType::InvalidConfiguration,
              "Invalid input configuration, could not retrieve calibration frames"};
    }

    std::shared_ptr<CvImage> cvinput;

    // Get the size of the current image
    const Core::InputDefinition& idef = videoInputs[camid];
    const int width = (int)idef.getWidth();
    const int height = (int)idef.getHeight();

    FAIL_RETURN(loadInputImage(cvinput, it->second, width, height));

    rig[camid].push_back(cvinput);
  }

  return Status::OK();
}

Potential<Ptv::Value> CalibrationOnlineAlgorithm::onFrame(
    Core::PanoDefinition& pano, std::vector<std::pair<videoreaderid_t, GPU::Surface&>>& frames, mtime_t /*date*/,
    FrameRate /*frameRate*/, Util::OpaquePtr** ctx) {
  /*Validate configuration*/
  if (!calibConfig.isValid()) {
    // TODOLATERSTATUS get output from CalibrationConfig parsing
    return {Origin::CalibrationAlgorithm, ErrType::InvalidConfiguration, "Invalid calibration configuration"};
  }

  if (calibConfig.getRigPreset()->getRigCameraDefinitionCount() != (size_t)pano.numVideoInputs()) {
    return {Origin::CalibrationAlgorithm, ErrType::InvalidConfiguration,
            "Calibration camera presets not matching the number of video inputs"};
  }

  /*
   * Retrieve the control point manager, passed as an opaque pointer from the client
   * or create a new one if no cp manager was passed
   */
  CalibrationProgress calibProgress(nullptr, getProgressUnits(pano.numVideoInputs(), 1));

  /*Perform calibration*/
  std::unique_ptr<Calibration, std::function<void(Calibration*)>> calibrationAlgorithm;
  if (ctx == nullptr) {
    calibrationAlgorithm = std::unique_ptr<Calibration, std::function<void(Calibration*)>>(
        new Calibration(calibConfig, calibProgress), [](Calibration* data) { delete data; });
  } else {
    if (*ctx == nullptr) {
      *ctx = new Calibration(calibConfig, calibProgress);
    }
    calibrationAlgorithm = std::unique_ptr<Calibration, std::function<void(Calibration*)>>(
        dynamic_cast<Calibration*>(*ctx), [](Calibration*) {});
  }

  RigCvImages rig;
  if (!calibConfig.isApplyingPresetsOnly()) {
    /*Load images onto cpu*/
    FAIL_RETURN(retrieveImages(rig, pano, frames));
  }

  return calibrationAlgorithm->process(pano, rig);
}

}  // namespace Calibration
}  // namespace VideoStitch
