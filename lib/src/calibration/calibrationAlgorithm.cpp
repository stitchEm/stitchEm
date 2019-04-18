// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationAlgorithm.hpp"

#include "cvImage.hpp"
#include "calibrationProgress.hpp"
#include "calibration.hpp"
#include "calibrationUtils.hpp"

#include "util/registeredAlgo.hpp"
#include "core/controllerInputFrames.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/rigDef.hpp"
#include "libvideostitch/rigCameraDef.hpp"

#define DEBUG_CALIBRATION_DUMP_FRAMES 0

#if DEBUG_CALIBRATION_DUMP_FRAMES

#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif

#include "util/pngutil.hpp"
#endif

namespace VideoStitch {
namespace Calibration {

namespace {
Util::RegisteredAlgo<CalibrationAlgorithm> registered("calibration");
}

const char* CalibrationAlgorithm::docString =
    "An algorithm that calibrates a panoramic multi-camera system using overlap zones between images\n";

CalibrationAlgorithm::CalibrationAlgorithm(const Ptv::Value* config) : CalibrationAlgorithmBase(config) {}

CalibrationAlgorithm::~CalibrationAlgorithm() {}

Potential<Ptv::Value> CalibrationAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                                                  Util::OpaquePtr**) const {
  if (!calibConfig.isValid()) {
    // TODOLATERSTATUS get output from CalibrationConfig parsing
    return {Origin::CalibrationAlgorithm, ErrType::InvalidConfiguration, "Invalid calibration configuration"};
  }

  if (calibConfig.getRigPreset()->getRigCameraDefinitionCount() != (size_t)pano->numVideoInputs()) {
    return {Origin::CalibrationAlgorithm, ErrType::InvalidConfiguration,
            "Calibration camera presets not matching the number of video inputs"};
  }

  CalibrationProgress calibProgress(
      progress, getProgressUnits(pano->numVideoInputs(), static_cast<int>(calibConfig.getFrames().size())));
  Calibration calib(calibConfig, calibProgress);
  RigCvImages rig;

  /*Extract images only if not applying only the presets*/
  if (!calibConfig.isApplyingPresetsOnly()) {
    FAIL_RETURN(retrieveImages(rig, *pano, calibProgress));
  }

  FAIL_RETURN(calib.process(*pano, rig));

  return calibProgress.add(CalibrationProgress::optim_done, "Calibration done");
}

Status CalibrationAlgorithm::retrieveImages(RigCvImages& rig, const Core::PanoDefinition& pano,
                                            CalibrationProgress& progress) const {
  /*Create rig of n list*/
  rig.clear();
  rig.resize(pano.numVideoInputs());

  auto container = Core::ControllerInputFrames<PixelFormat::Grayscale, unsigned char>::create(&pano);
  FAIL_RETURN(container.status());

  for (auto& numFrame : calibConfig.getFrames()) {
    std::map<readerid_t, PotentialValue<GPU::HostBuffer<unsigned char>>> loadedFrames;

    FAIL_RETURN(container->seek((int)numFrame));
    FAIL_RETURN(container->load(loadedFrames));

    for (auto& loadedFrame : loadedFrames) {
      readerid_t inputid = loadedFrame.first;

      if (inputid >= (int)pano.numInputs()) {
        return {Origin::CalibrationAlgorithm, ErrType::InvalidConfiguration,
                "Invalid input configuration, could not load calibration frames"};
      }

      FAIL_RETURN(progress.add(CalibrationProgress::seek, "Seeking frames"));

      auto potLoadedFrame = loadedFrames.at(inputid);
      FAIL_RETURN(potLoadedFrame.status());

      GPU::HostBuffer<unsigned char> frame = potLoadedFrame.value();

      /*Get the size of the current image*/
      const Core::InputDefinition& idef = pano.getInput(inputid);
      const int width = (int)idef.getWidth();
      const int height = (int)idef.getHeight();

      auto potHostFrame = GPU::HostBuffer<unsigned char>::allocate(frame.numElements(), "Calibration frame loading");
      FAIL_RETURN(potHostFrame.status());
      GPU::HostBuffer<unsigned char> hostFrame = potHostFrame.value();
      std::memcpy(hostFrame.hostPtr(), frame.hostPtr(), frame.byteSize());

      std::shared_ptr<CvImage> cvinput(new CvImage(hostFrame, width, height));

#if DEBUG_CALIBRATION_DUMP_FRAMES
      std::ostringstream oss;
      oss << "calibration_frame_" << numFrame << "_" << inputid << ".png";
      Util::PngReader writer;
      writer.writeMonochromToFile(oss.str().c_str(), cvinput->cols, cvinput->rows, cvinput->data);
#endif

      /*Store the input pictures by videoinputid, not by inputid*/
      rig[pano.convertInputIndexToVideoInputIndex(inputid)].push_back(cvinput);
    }
  }

  return Status::OK();
}

}  // namespace Calibration
}  // namespace VideoStitch
