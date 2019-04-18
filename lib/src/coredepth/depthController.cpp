// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "depthControllerImpl.hpp"

#include "depthPipeline.hpp"

#include "core/readerController.hpp"
#include "core/stitchOutput/asyncOutput.hpp"

#include "libvideostitch/depthDef.hpp"
#include "libvideostitch/panoDef.hpp"

#include <sstream>

namespace VideoStitch {
namespace Core {

DepthControllerImpl::DepthControllerImpl(const PanoDefinition& pano, ReaderController* readerController,
                                         InputPipeline* pipe)
    : InputControllerImpl(readerController), pano(pano.clone()), pipe(pipe) {}

Potential<DepthController> DepthControllerImpl::create(const PanoDefinition& pano, const DepthDefinition& depthDef,
                                                       const AudioPipeDefinition& audioPipeDef,
                                                       Input::ReaderFactory* readerFactory) {
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

  auto potReaderController = ReaderController::create(pano, audioPipeDef, readerFactory);

  FAIL_RETURN(potReaderController.status());

#if USE_SGM
  auto potPipeline = SGMDepthPipeline::createSGMDepthPipeline(potReaderController->getReaders(), pano, depthDef);
#else
  auto potPipeline = DepthPipeline::createDepthPipeline(potReaderController->getReaders(), pano, depthDef);
#endif
  FAIL_RETURN(potPipeline.status());

  return new DepthControllerImpl(pano, potReaderController.release(), potPipeline.release());
}

DepthControllerImpl::~DepthControllerImpl() { delete pipe; }

ControllerStatus DepthControllerImpl::estimateDepth(std::vector<ExtractOutput*> extracts) {
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

Potential<ExtractOutput> DepthControllerImpl::createAsyncExtractOutput(
    int sourceID, std::shared_ptr<SourceSurface> surf, std::shared_ptr<VideoStitch::Output::VideoWriter> writer) {
  Potential<AsyncSourceOutput> potSourceOutput = AsyncSourceOutput::create({surf}, {}, {writer}, sourceID);
  FAIL_RETURN(potSourceOutput.status());
  return new ExtractOutput(potSourceOutput.release());
}

Potential<DepthController> createDepthController(const PanoDefinition& pano, const DepthDefinition& depthDef,
                                                 const AudioPipeDefinition& audioPipeDef,
                                                 Input::ReaderFactory* readerFactory) {
  return DepthControllerImpl::create(pano, depthDef, audioPipeDef, readerFactory);
}

}  // namespace Core
}  // namespace VideoStitch
