// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "undistortControllerImpl.hpp"

#include "undistortPipeline.hpp"

#include "core/readerController.hpp"
#include "core/stitchOutput/asyncOutput.hpp"

#include "libvideostitch/panoDef.hpp"

#include <sstream>

namespace VideoStitch {
namespace Core {

UndistortControllerImpl::UndistortControllerImpl(const PanoDefinition& pano, ReaderController* readerController,
                                                 InputPipeline* pipe, const OverrideOutputDefinition& outputDef)
    : InputControllerImpl(readerController), pano(pano.clone()), pipe(pipe), outputDef(outputDef) {}

Potential<UndistortController> UndistortControllerImpl::create(const PanoDefinition& pano,
                                                               const AudioPipeDefinition& audioPipeDef,
                                                               Input::ReaderFactory* readerFactory,
                                                               const OverrideOutputDefinition& outputDef) {
  {
    std::stringstream validationMessages;
    if (!pano.validate(validationMessages)) {
      return {Origin::Stitcher, ErrType::InvalidConfiguration,
              "Could not validate panorama configuration: " + validationMessages.str()};
    }
  }

  if (!pano.numInputs()) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration, "Configuration does not cotain any inputs"};
  }

  if (outputDef.manualFocal && outputDef.overrideFocal < 0.0) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration, "Trying to override focals with a negative value"};
  }

  auto potReaderController = ReaderController::create(pano, audioPipeDef, readerFactory);

  FAIL_RETURN(potReaderController.status());

  auto potPipeline = UndistortPipeline::createUndistortPipeline(potReaderController->getReaders(), pano, outputDef);
  FAIL_RETURN(potPipeline.status());

  return new UndistortControllerImpl(pano, potReaderController.release(), potPipeline.release(), outputDef);
}

UndistortControllerImpl::~UndistortControllerImpl() { delete pipe; }

ControllerStatus UndistortControllerImpl::undistort(std::vector<ExtractOutput*> extracts) {
  auto statusVideo = Input::ReadStatus::fromCode<Input::ReadStatusCode::EndOfFile>();
  auto statusAudio = Input::ReadStatus::fromCode<Input::ReadStatusCode::EndOfFile>();
  auto statusMetadata = Input::ReadStatus::fromCode<Input::ReadStatusCode::EndOfFile>();

  // load the acquisition data
  std::map<readerid_t, Input::PotentialFrame> inputBuffers;
  mtime_t date;
  std::vector<Audio::audioBlockGroupMap_t> audioBlocks;
  Input::MetadataChunk metadata;

  std::tie(statusVideo, statusAudio, statusMetadata) =
      readerController->load(date, inputBuffers, audioBlocks, metadata);

  if (statusVideo.ok()) {
    pipe->process(date, getFrameRate(), inputBuffers, extracts);
  }

  readerController->releaseBuffer(inputBuffers);

  switch (statusVideo.getCode()) {
    case Input::ReadStatusCode::ErrorWithStatus:
      return statusVideo.getStatus();
    case Input::ReadStatusCode::TryAgain:
      return Status{Origin::Stitcher, ErrType::RuntimeError, "Couldn't load inputs (TryAgain)"};
    case Input::ReadStatusCode::EndOfFile:
      return ControllerStatus::fromCode<ControllerStatusCode::EndOfStream>();
    case Input::ReadStatusCode::Ok:
      break;
  }

  return Status::OK();
}

Potential<ExtractOutput> UndistortControllerImpl::createAsyncExtractOutput(
    int sourceID, std::shared_ptr<SourceSurface> surf, std::shared_ptr<VideoStitch::Output::VideoWriter> writer) {
  Potential<AsyncSourceOutput> potSourceOutput = AsyncSourceOutput::create({surf}, {}, {writer}, sourceID);
  FAIL_RETURN(potSourceOutput.status());
  return new ExtractOutput(potSourceOutput.release());
}

Potential<PanoDefinition> UndistortControllerImpl::createPanoDefWithoutDistortion() {
  // clone pano
  std::unique_ptr<PanoDefinition> newPano{pano->clone()};

  // remove distortion on all inputs
  for (Core::InputDefinition& idef : newPano->getVideoInputs()) {
    if (idef.hasCroppedArea()) {
      Logger::get(Logger::Info, "UndistortController")
          << "Dropping crop values to create pano definition without distortion" << std::endl;
      idef.resetCrop();
    }

    if (idef.getMaskPixelDataIfValid() != nullptr) {
      return Status{Origin::Input, ErrType::InvalidConfiguration,
                    "Masks are not preserved during undistortion, cannot create new PanoDefinition"};
    }

    outputDef.applyOverrideSettings(idef);

    idef.resetDistortion();

    idef.resetExposureValue();
    idef.resetRedCB();
    idef.resetGreenCB();
    idef.resetBlueCB();
    idef.resetVignetting();
    idef.resetPhotoResponse();
  }

  if (newPano->hasBeenCalibrated()) {
    Logger::get(Logger::Warning, "UndistortController")
        << "Previous calibration found. Calibration may become invalid in new PanoDefinition with undistorted inputs."
        << std::endl;
  }

  return newPano.release();
}

Potential<UndistortController> createUndistortController(const PanoDefinition& pano,
                                                         const AudioPipeDefinition& audioPipeDef,
                                                         Input::ReaderFactory* readerFactory,
                                                         const OverrideOutputDefinition& outputDef) {
  return UndistortControllerImpl::create(pano, audioPipeDef, readerFactory, outputDef);
}

}  // namespace Core
}  // namespace VideoStitch
