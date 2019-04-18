// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "controller.hpp"

#include "panoPipeline.hpp"
#include "stereoPipeline.hpp"

#include "stitchOutput/asyncOutput.hpp"
#include "stitchOutput/blockingOutput.hpp"
#include "libvideostitch/orah/imuStabilization.hpp"
#include "core1/panoStitcher.hpp"
#include "coredepth/depthStitcher.hpp"
#include "audio/asrc.hpp"
#include "audio/audioPipeFactory.hpp"

#include "libvideostitch/audioBlock.hpp"
#include "libvideostitch/config.hpp"
#include "libvideostitch/imageMergerFactory.hpp"
#include "libvideostitch/imageWarperFactory.hpp"
#include "libvideostitch/imageFlowFactory.hpp"
#include "libvideostitch/orah/exposureData.hpp"
#include "libvideostitch/stereoRigDef.hpp"

#include <algorithm>
#include <utility>

namespace VideoStitch {
namespace Core {

static std::chrono::milliseconds UPDATE_PANORAMA_TIMEOUT{16};

// -------------------------- Controller ---------------------------

template <typename VideoPipeline>
ControllerImpl<VideoPipeline>::ControllerImpl(const PanoDefinition& pano, Audio::AudioPipeline* audioPipe,
                                              const ImageMergerFactory& mergerFactory,
                                              const ImageWarperFactory& warperFactory,
                                              const ImageFlowFactory& flowFactory, ReaderController* readerController,
                                              std::vector<PreProcessor*> preprocessors, PostProcessor* postprocessor,
                                              const StereoRigDefinition* rig)
    : VideoStitch::Core::InputControllerImpl(readerController),
      pano(pano.clone()),
      rig(rig ? rig->clone() : nullptr),
      mergerFactory(mergerFactory.clone()),
      warperFactory(warperFactory.clone()),
      flowFactory(flowFactory.clone()),
      setupPending(false),
      preprocessors(std::move(preprocessors)),
      preProcessingEnabled(true),
      metadataProcessingEnabled(false),
      postprocessor(postprocessor),
      audioPipe(audioPipe),
      videoPipe(nullptr),
      stabilizationEnabled(false) {}

template <typename VideoPipeline>
auto ControllerImpl<VideoPipeline>::create(const PanoDefinition& pano, const AudioPipeDefinition& audioPipeDef,
                                           const ImageMergerFactory& mergerFactory,
                                           const ImageWarperFactory& warperFactory, const ImageFlowFactory& flowFactory,
                                           Input::ReaderFactory* readerFactory, const StereoRigDefinition* rig)
    -> PotentialController {
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

  // Video pre-processors
  std::vector<PreProcessor*> preprocessors;
  for (readerid_t imId = 0; imId < pano.numInputs(); ++imId) {
    if (pano.getInput(imId).getPreprocessors()) {
      auto potPP = PreProcessor::create(*pano.getInput(imId).getPreprocessors());
      if (!potPP.ok()) {
        Logger::get(Logger::Error) << "Error: Cannot create preprocessor for input " << imId
                                   << ". Trying to continue anyway..." << std::endl;
      }
      preprocessors.push_back(potPP.release());
    } else {
      preprocessors.push_back(nullptr);
    }
  }

  // post-processor
  PostProcessor* postprocessor = nullptr;
  if (pano.getPostprocessors()) {
    Potential<PostProcessor> potential = PostProcessor::create(*pano.getPostprocessors());
    if (!potential.ok()) {
      Logger::get(Logger::Error) << "Error: Cannot create postprocessor. Trying to continue anyway..." << std::endl;
    }
    postprocessor = potential.release();
  }

  // Audio Pipeline
  Audio::AudioPipeline* audioPipe = Audio::AudioPipeFactory::create(audioPipeDef, pano).release();

  auto ctrl = new ControllerImpl(pano, audioPipe, mergerFactory, warperFactory, flowFactory,
                                 potReaderController.release(), std::move(preprocessors), postprocessor, rig);

  return PotentialController(ctrl);
}

template <typename VideoPipeline>
ControllerImpl<VideoPipeline>::~ControllerImpl() {
  delete pano;
  delete rig;
  // delete the owned preprocessors
  for (unsigned i = 0; i < preprocessors.size(); ++i) {
    delete preprocessors[i];
  }
  if (postprocessor) {
    delete postprocessor;
  }
  delete mergerFactory;
  delete warperFactory;
  delete flowFactory;
  delete videoPipe;
  delete audioPipe;
}

PotentialController createController(const PanoDefinition& pano, const ImageMergerFactory& mergerFactory,
                                     const ImageWarperFactory& warperFactory, const ImageFlowFactory& flowFactory,
                                     Input::ReaderFactory* readerFactory, const AudioPipeDefinition& audioPipe) {
  return ControllerImpl<PanoPipeline>::create(pano, audioPipe, mergerFactory, warperFactory, flowFactory, readerFactory,
                                              nullptr);
}

void deleteController(Controller* controller) {
  if (controller == nullptr) {
    return;
  }
  ControllerImpl<PanoPipeline>* controllerImpl = static_cast<ControllerImpl<PanoPipeline>*>(controller);
  delete controllerImpl;
}

PotentialStereoController createController(const PanoDefinition& pano, const StereoRigDefinition& rig,
                                           const ImageMergerFactory& mergerFactory,
                                           const ImageWarperFactory& warperFactory, const ImageFlowFactory& flowFactory,
                                           Input::ReaderFactory* readerFactory) {
  std::unique_ptr<AudioPipeDefinition> uniqueAudioPipeDef(AudioPipeDefinition::createDefault());
  return ControllerImpl<StereoPipeline>::create(pano, *uniqueAudioPipeDef, mergerFactory, warperFactory, flowFactory,
                                                readerFactory, &rig);
}

void deleteController(StereoController* controller) {
  if (controller == nullptr) {
    return;
  }
  ControllerImpl<StereoPipeline>* controllerImpl = static_cast<ControllerImpl<StereoPipeline>*>(controller);
  delete controllerImpl;
}

template <typename VideoPipeline>
bool ControllerImpl<VideoPipeline>::isPanoChangeCompatible(const PanoDefinition& newPano) const {
  // FIXME: (maybe) Make sure stitchers are not stitching by locking;
  switch (videoPipe->getCompatibility(*pano, newPano)) {
    case IncompatibleChanges:
      return false;
    default:
      break;
  }
  return true;
}

template <typename VideoPipeline>
Status ControllerImpl<VideoPipeline>::resetPano(const PanoDefinition& newPano) {
  {
    std::stringstream validationMessages;
    if (!newPano.validate(validationMessages)) {
      return {Origin::Stitcher, ErrType::InvalidConfiguration,
              "New panorama configuration is invalid: " + validationMessages.str()};
    }
  }

  // FIXME: (maybe) Make sure stitchers are not stitching by locking;
  if (videoPipe != nullptr) {
    switch (videoPipe->getCompatibility(*pano, newPano)) {
      case IncompatibleChanges:
        return {Origin::Stitcher, ErrType::InvalidConfiguration, "Setup changes are incompatible"};
      case SetupIncompatibleChanges:
        setupPending = true;
        break;
      case SetupCompatibleChanges:
        break;
    }
  }
  PanoDefinition* myOldPano = pano;
  pano = newPano.clone();
  delete myOldPano;

  readerController->resetPano(newPano);

  if (videoPipe != nullptr) {
    if (setupPending) {
      PROPAGATE_FAILURE_STATUS(videoPipe->redoSetup(*pano, *mergerFactory, *warperFactory, *flowFactory, rig));
    } else {
      videoPipe->setPano(*pano);
    }
  }
  audioPipe->resetPano(*pano);
  setupPending = false;
  return Status::OK();
}

template <typename VideoPipeline>
Status ControllerImpl<VideoPipeline>::updatePanorama(
    const std::function<Potential<PanoDefinition>(const PanoDefinition&)>& panoramaUpdater) {
  std::unique_lock<std::timed_mutex> lock(panoramaUpdateLock, std::defer_lock);
  if (!lock.try_lock_for(UPDATE_PANORAMA_TIMEOUT)) {
    return Status(Origin::PanoramaConfiguration, ErrType::ImplementationError, "Panorama update timeout");
  }

  auto panorama = panoramaUpdater(getPano());
  if (!panorama.ok()) {
    return panorama.status();
  }
  // todo: consider shared lock with getpano?
  PROPAGATE_FAILURE_STATUS(resetPano(*panorama.object()));
  return Status();
}

template <typename VideoPipeline>
Status ControllerImpl<VideoPipeline>::updatePanorama(const PanoDefinition& panorama) {
  auto panoramaPointer = panorama.clone();
  return updatePanorama(
      [panoramaPointer](const PanoDefinition&) { return Potential<PanoDefinition>(panoramaPointer); });
}

template <typename VideoPipeline>
Status ControllerImpl<VideoPipeline>::applyAudioProcessorParam(const AudioPipeDefinition& newAudioPipe) {
  return audioPipe->applyProcessorParam(newAudioPipe);
}

template <typename VideoPipeline>
Status ControllerImpl<VideoPipeline>::setAudioDelay(double delay_ms) {
  return audioPipe->setDelay(delay_ms / 1000.);
}

template <typename VideoPipeline>
Status ControllerImpl<VideoPipeline>::resetRig(const StereoRigDefinition& newRig) {
  delete rig;
  rig = newRig.clone();
  PROPAGATE_FAILURE_STATUS(videoPipe->redoSetup(*pano, *mergerFactory, *warperFactory, *flowFactory, rig));
  return Status::OK();
}

template <typename VideoPipeline>
Status ControllerImpl<VideoPipeline>::resetMergerFactory(const ImageMergerFactory& newMergerFactory,
                                                         bool redoSetupNow) {
  if (mergerFactory->equal(newMergerFactory)) {
    return Status::OK();
  }
  delete mergerFactory;
  mergerFactory = newMergerFactory.clone();
  if (videoPipe != nullptr && redoSetupNow) {
    PROPAGATE_FAILURE_STATUS(videoPipe->redoSetup(*pano, *mergerFactory, *warperFactory, *flowFactory, rig));
    setupPending = false;
  } else {
    setupPending = true;
  }
  return Status::OK();
}

template <typename VideoPipeline>
Status ControllerImpl<VideoPipeline>::resetWarperFactory(const ImageWarperFactory& newWarperFactory,
                                                         bool redoSetupNow) {
  if (warperFactory->equal(newWarperFactory)) {
    return Status::OK();
  }
  delete warperFactory;
  warperFactory = newWarperFactory.clone();
  if (redoSetupNow) {
    PROPAGATE_FAILURE_STATUS(videoPipe->redoSetup(*pano, *mergerFactory, *warperFactory, *flowFactory, rig));
    setupPending = false;
  } else {
    setupPending = true;
  }
  return Status::OK();
}

template <typename VideoPipeline>
Status ControllerImpl<VideoPipeline>::resetFlowFactory(const ImageFlowFactory& newFlowFactory, bool redoSetupNow) {
  if (flowFactory->equal(newFlowFactory)) {
    return Status::OK();
  }
  delete flowFactory;
  flowFactory = newFlowFactory.clone();
  if (redoSetupNow) {
    PROPAGATE_FAILURE_STATUS(videoPipe->redoSetup(*pano, *mergerFactory, *warperFactory, *flowFactory, rig));
    setupPending = false;
  } else {
    setupPending = true;
  }
  return Status::OK();
}

template <typename VideoPipeline>
PreProcessor* ControllerImpl<VideoPipeline>::getPreProcessor(int i) const {
  assert(i < (int)preprocessors.size());
  return preprocessors[i];
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::setPreProcessor(int i, PreProcessor* p) {
  assert(i < (int)preprocessors.size());
  if (preprocessors[i]) {
    delete preprocessors[i];
  }
  preprocessors[i] = p;

  if (preProcessingEnabled) {
    videoPipe->setPreProcessors(preprocessors);
  }
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::enablePreProcessing(bool value) {
  preProcessingEnabled = value;

  if (preProcessingEnabled) {
    videoPipe->setPreProcessors(preprocessors);
  } else {
    std::vector<PreProcessor*> nullptr_array(preprocessors.size(), nullptr);
    videoPipe->setPreProcessors(nullptr_array);
  }
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::enableMetadataProcessing(bool value) {
  metadataProcessingEnabled = value;
}

template <typename VideoPipeline>
PostProcessor* ControllerImpl<VideoPipeline>::getPostProcessor() const {
  return postprocessor;
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::setPostProcessor(PostProcessor* p) {
  if (postprocessor) {
    delete postprocessor;
  }
  postprocessor = p;
  videoPipe->setPostProcessor(postprocessor);
}

template <typename VideoPipeline>
bool ControllerImpl<VideoPipeline>::hasVuMeter(const std::string& inputName) const {
  return audioPipe->hasVuMeter(inputName);
}

template <typename VideoPipeline>
std::vector<double> ControllerImpl<VideoPipeline>::getPeakValues(const std::string& inputName) const {
  PotentialValue<std::vector<double>> ret = audioPipe->getPeakValues(inputName);
  if (ret.ok()) {
    return ret.value();
  }

  return {};
}

template <typename VideoPipeline>
std::vector<double> ControllerImpl<VideoPipeline>::getRMSValues(const std::string& inputName) const {
  PotentialValue<std::vector<double>> ret = audioPipe->getRMSValues(inputName);
  if (ret.ok()) {
    return ret.value();
  }

  return {};
}

template <typename VideoPipeline>
bool ControllerImpl<VideoPipeline>::addAudioOutput(std::shared_ptr<VideoStitch::Output::AudioWriter> o) {
  return audioPipe->addOutput(o);
}

template <typename VideoPipeline>
bool ControllerImpl<VideoPipeline>::removeAudioOutput(const std::string& id) {
  return audioPipe->removeOutput(id);
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::setAudioInput(const std::string& inputName) {
  return audioPipe->setInput(inputName);
}

// ------------------------- Stitcher factory -----------------------

Potential<StereoPipeline> makeStitcher(ControllerImpl<StereoPipeline>& controller,
                                       ImageMergerFactory::CoreVersion version) {
  StereoPipeline* ret = nullptr;

  // left device is used for input extraction
  switch (version) {
    case ImageMergerFactory::CoreVersion1:
      return StereoPipeline::createStereoPipeline(
          new PanoStitcherImplV1<StereoOutput>("left", controller.getPano(), LeftEye),
          new PanoStitcherImplV1<StereoOutput>("right", controller.getPano(), RightEye),
          controller.getReaderCtrl().getReaders(), controller.getPreProcessors(), controller.getPostProcessor());
      break;
    case ImageMergerFactory::Depth:
      // Depth + Stereo not supported
      return nullptr;
    case ImageMergerFactory::Impotent:
      return nullptr;  // XXX TODO FIXME
  }

  return ret;
}
Potential<PanoPipeline> makeStitcher(ControllerImpl<PanoPipeline>& controller,
                                     ImageMergerFactory::CoreVersion version) {
  switch (version) {
    case ImageMergerFactory::CoreVersion1:
      return PanoPipeline::createPanoPipeline(
          new PanoStitcherImplV1<StitchOutput>("pano", controller.getPano(), LeftEye),
          controller.getReaderCtrl().getReaders(), controller.getPreProcessors(), controller.getPostProcessor());
    case ImageMergerFactory::Depth:
      return PanoPipeline::createPanoPipeline(new DepthStitcher<StitchOutput>("depth", controller.getPano(), LeftEye),
                                              controller.getReaderCtrl().getReaders(), controller.getPreProcessors(),
                                              controller.getPostProcessor());
    case ImageMergerFactory::Impotent:
    default:
      return nullptr;  // XXX TODO FIXME
  }
}

template <typename VideoPipeline>
Status ControllerImpl<VideoPipeline>::createStitcher() {
  std::unique_lock<std::mutex> lock(stitcherMutex);
  std::unique_ptr<VideoPipeline> stitcher{nullptr};

  FAIL_RETURN(getReaderCtrl().setupReaders());

  auto potStitcher = makeStitcher(*this, mergerFactory->version());
  FAIL_RETURN(potStitcher.status());
  stitcher.reset(potStitcher.release());

  // TODO: do not redo the setup each time.
  const Status stitcherSetupStatus = stitcher->setup(*mergerFactory, *warperFactory, *flowFactory, rig);
  if (!stitcherSetupStatus.ok()) {
    readerController->cleanReaders();
  }
  FAIL_RETURN(stitcherSetupStatus);

  delete videoPipe;
  videoPipe = stitcher.release();
  return Status::OK();
}

void makeDefaultDevice(StereoDeviceDefinition& def) {
  def.leftDevice = 0;
  def.rightDevice = 0;
}
void makeDefaultDevice(PanoDeviceDefinition& def) { def.device = 0; }

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::deleteStitcher() {
  std::unique_lock<std::mutex> lock(stitcherMutex);
  readerController->cleanReaders();
  delete videoPipe;
  videoPipe = nullptr;
}

// -------------------------- Outputs factory : sources -------------------

namespace {
Potential<ExtractOutput> makeBlockingExtractOutput(int source, std::shared_ptr<SourceSurface> surf,
                                                   const std::vector<std::shared_ptr<SourceRenderer>>& renderers,
                                                   const std::vector<std::shared_ptr<Output::VideoWriter>>& writers) {
  Potential<BlockingSourceOutput> potentialSourceOutput =
      BlockingSourceOutput::create(surf, renderers, writers, source);
  if (!potentialSourceOutput.ok()) {
    return potentialSourceOutput.status();
  }
  return new ExtractOutput(potentialSourceOutput.release());
}

}  // namespace

template <typename VideoPipeline>
Potential<ExtractOutput> ControllerImpl<VideoPipeline>::createBlockingExtractOutput(
    int source, std::shared_ptr<SourceSurface> surf, std::shared_ptr<SourceRenderer> renderer,
    std::shared_ptr<VideoStitch::Output::VideoWriter> writer) {
  std::vector<std::shared_ptr<VideoStitch::Output::VideoWriter>> writers;
  if (writer) {
    writers.push_back(writer);
  }
  std::vector<std::shared_ptr<SourceRenderer>> renderers;
  if (renderer) {
    renderers.push_back(renderer);
  }
  return makeBlockingExtractOutput(source, surf, renderers, writers);
}

namespace {
Potential<ExtractOutput> makeAsyncExtractOutput(int source, const std::vector<std::shared_ptr<SourceSurface>>& surf,
                                                const std::vector<std::shared_ptr<SourceRenderer>>& renderers,
                                                const std::vector<std::shared_ptr<Output::VideoWriter>>& writers) {
  Potential<AsyncSourceOutput> potentialSourceOutput = AsyncSourceOutput::create(surf, renderers, writers, source);
  if (!potentialSourceOutput.ok()) {
    return potentialSourceOutput.status();
  }
  return new ExtractOutput(potentialSourceOutput.release());
}

}  // namespace

template <typename VideoPipeline>
Potential<ExtractOutput> ControllerImpl<VideoPipeline>::createAsyncExtractOutput(
    int source, const std::vector<std::shared_ptr<SourceSurface>>& surf, std::shared_ptr<SourceRenderer> renderer,
    std::shared_ptr<VideoStitch::Output::VideoWriter> writer) const {
  std::vector<std::shared_ptr<VideoStitch::Output::VideoWriter>> writers;
  if (writer) {
    writers.push_back(writer);
  }
  std::vector<std::shared_ptr<SourceRenderer>> renderers;
  if (renderer) {
    renderers.push_back(renderer);
  }
  return makeAsyncExtractOutput(source, surf, renderers, writers);
}

// -------------------------- Outputs factory : panoramas -------------------

namespace {
Potential<StitchOutput> makeBlockingOutput(std::shared_ptr<PanoSurface> surf,
                                           const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                                           const std::vector<std::shared_ptr<Output::VideoWriter>>& writers) {
  Potential<BlockingStitchOutput> potentialStitchOutput = BlockingStitchOutput::create(surf, renderers, writers);
  if (!potentialStitchOutput.ok()) {
    return potentialStitchOutput.status();
  }
  return new StitchOutput(potentialStitchOutput.release());
}
Potential<StereoOutput> makeBlockingOutput(std::shared_ptr<PanoSurface> surf,
                                           const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                                           const std::vector<std::shared_ptr<Output::StereoWriter>>& writers) {
  Potential<BlockingStereoOutput> potentialStitchOutput = BlockingStereoOutput::create(surf, renderers, writers);
  if (!potentialStitchOutput.ok()) {
    return potentialStitchOutput.status();
  }
  return new StereoOutput(potentialStitchOutput.release());
}
}  // namespace

template <typename VideoPipeline>
auto ControllerImpl<VideoPipeline>::createBlockingStitchOutput(
    std::shared_ptr<PanoSurface> surf, const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
    const std::vector<std::shared_ptr<Writer>>& writers) -> PotentialOutput {
  return PotentialOutput(makeBlockingOutput(surf, renderers, writers));
}

namespace {
Potential<StitchOutput> makeAsyncOutput(const std::vector<std::shared_ptr<PanoSurface>>& surf,
                                        const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                                        const std::vector<std::shared_ptr<Output::VideoWriter>>& writers) {
  Potential<AsyncStitchOutput> potentialStitchOutput = AsyncStitchOutput::create(surf, renderers, writers);
  if (!potentialStitchOutput.ok()) {
    return potentialStitchOutput.status();
  }
  return new StitchOutput(potentialStitchOutput.release());
}
Potential<StereoOutput> makeAsyncOutput(const std::vector<std::shared_ptr<PanoSurface>>& surf,
                                        const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                                        const std::vector<std::shared_ptr<Output::StereoWriter>>& writers) {
  Potential<AsyncStereoOutput> potentialStitchOutput = AsyncStereoOutput::create(surf, renderers, writers);
  if (!potentialStitchOutput.ok()) {
    return potentialStitchOutput.status();
  }
  return new StereoOutput(potentialStitchOutput.release());
}
}  // namespace

template <typename VideoPipeline>
auto ControllerImpl<VideoPipeline>::createAsyncStitchOutput(const std::vector<std::shared_ptr<PanoSurface>>& surf,
                                                            const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                                                            const std::vector<std::shared_ptr<Writer>>& writers) const
    -> PotentialOutput {
  return PotentialOutput(makeAsyncOutput(surf, renderers, writers));
}

// ------------------ stitching interface --------------

template <typename VideoPipeline>
ControllerStatus ControllerImpl<VideoPipeline>::stitch(Output* output, bool readFrame) {
  std::vector<ExtractOutput*> ext;
  return stitchAndExtract(output, ext, nullptr, readFrame);
}

ControllerStatus videoLoadStatus(const Input::ReadStatus& videoLoadStatus) {
  switch (videoLoadStatus.getCode()) {
    case Input::ReadStatusCode::Ok:
      return ControllerStatus::OK();
    case Input::ReadStatusCode::ErrorWithStatus:
      return ControllerStatus::fromError(
          {Origin::Input, ErrType::RuntimeError, "Could not load input frames", videoLoadStatus.getStatus()});
    case Input::ReadStatusCode::EndOfFile:
      return ControllerStatus::fromCode<ControllerStatusCode::EndOfStream>();
    case Input::ReadStatusCode::TryAgain:
      return ControllerStatus::fromError(
          {Origin::Input, ErrType::RuntimeError, "Could not load input frames, reader starved"});
  }
  assert(false);
  return ControllerStatus::fromError(
      {Origin::Input, ErrType::ImplementationError, "Could not load input frames, unknown error code"});
}

template <typename VideoPipeline>
ControllerStatus ControllerImpl<VideoPipeline>::extract(ExtractOutput* extract, bool readFrame) {
  // load the acquisition data
  std::map<readerid_t, Input::PotentialFrame> inputBuffers;
  std::vector<Audio::audioBlockGroupMap_t> audioBlocks;
  Input::MetadataChunk metadata;

  mtime_t date;
  if (readFrame) {
    auto loadStatus = readerController->load(date, inputBuffers, audioBlocks, metadata);
    FAIL_CONTROLLER_RETURN(videoLoadStatus(std::get<0>(loadStatus)));
  } else {
    date = readerController->reload(inputBuffers);
    if (inputBuffers.size() == 0) {
      return ControllerStatus::fromError({Origin::Input, ErrType::RuntimeError, "Could not reload input frames"});
    }
  }

  Status extractStatus = videoPipe->extract(date, inputBuffers, extract);
  readerController->releaseBuffer(inputBuffers);

  return ControllerStatus::fromError(extractStatus);
}

template <typename VideoPipeline>
ControllerStatus ControllerImpl<VideoPipeline>::extract(std::vector<ExtractOutput*> extracts, AlgorithmOutput* algo,
                                                        bool readFrame) {
  // load the acquisition data
  std::map<readerid_t, Input::PotentialFrame> inputBuffers;
  std::vector<Audio::audioBlockGroupMap_t> audioBlocks;
  Input::MetadataChunk metadata;

  mtime_t date;
  if (readFrame) {
    auto loadStatus = readerController->load(date, inputBuffers, audioBlocks, metadata);
    FAIL_CONTROLLER_RETURN(videoLoadStatus(std::get<0>(loadStatus)));
  } else {
    date = readerController->reload(inputBuffers);
    if (inputBuffers.size() == 0) {
      return ControllerStatus::fromError({Origin::Input, ErrType::RuntimeError, "Could not reload input frames"});
    }
  }

  Status extractStatus = videoPipe->extract(date, readerController->getFrameRate(), inputBuffers, extracts, algo);
  readerController->releaseBuffer(inputBuffers);

  return ControllerStatus::fromError(extractStatus);
}

bool isSamplesEmpty(std::map<readerid_t, Audio::AudioBlock>& samples) {
  for (auto& kv : samples) {
    if (kv.second.size() > 0) {
      return false;
    }
  }
  return true;
}

template <typename VideoPipeline>
const Quaternion<double> ControllerImpl<VideoPipeline>::getUserOrientation() {
  return qUserOrientation;
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::setUserOrientation(const Quaternion<double>& q) {
  qUserOrientation = q;
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::updateUserOrientation(const Quaternion<double>& q) {
  qUserOrientation *= q;
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::resetUserOrientation() {
  setUserOrientation(Quaternion<double>(1., 0., 0., 0));
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::enableStabilization(bool value) {
  resetUserOrientation();
  stabilizationEnabled = value;
}

template <typename VideoPipeline>
bool ControllerImpl<VideoPipeline>::isStabilizationEnabled() {
  return stabilizationEnabled;
}

template <typename VideoPipeline>
Stab::IMUStabilization& ControllerImpl<VideoPipeline>::getStabilizationIMU() {
  return stabilizationAlgorithm;
}

template <typename VideoPipeline>
mtime_t ControllerImpl<VideoPipeline>::getLatency() const {
  return readerController->getLatency();
}

template <typename VideoPipeline>
Status ControllerImpl<VideoPipeline>::addSink(const Ptv::Value* config) {
  return readerController->addSink(config);
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::removeSink() {
  return readerController->removeSink();
}

template <typename VideoPipeline>
ControllerStatus ControllerImpl<VideoPipeline>::stitchAndExtract(Output* output, std::vector<ExtractOutput*> extracts,
                                                                 AlgorithmOutput* algo, bool readFrame) {
  auto statusVideo = Input::ReadStatus::fromCode<Input::ReadStatusCode::EndOfFile>();
  auto statusAudio = Input::ReadStatus::fromCode<Input::ReadStatusCode::EndOfFile>();
  auto statusMetadata = Input::ReadStatus::fromCode<Input::ReadStatusCode::EndOfFile>();

  // load the acquisition data
  std::map<readerid_t, Input::PotentialFrame> inputBuffers;
  mtime_t date;
  std::vector<Audio::audioBlockGroupMap_t> audioBlocks;
  Input::MetadataChunk metadata;

  if (readFrame) {
    std::tie(statusVideo, statusAudio, statusMetadata) =
        readerController->load(date, inputBuffers, audioBlocks, metadata);
  } else {
    date = readerController->reload(inputBuffers);
    if (inputBuffers.size() == 0) {
      return ControllerStatus::fromError({Origin::Input, ErrType::RuntimeError, "Could not reload input frames"});
    } else {
      statusVideo = Input::ReadStatus::OK();
    }
  }

  // process audio
  if (statusAudio.ok() || statusAudio.getCode() == Input::ReadStatusCode::TryAgain) {
    for (auto& samples : audioBlocks) {
      if (!samples.empty()) {
        audioPipe->process(samples);
      }
    }
  }

  FAIL_CONTROLLER_RETURN(videoLoadStatus(statusVideo));

  Status videoPipeStatus;

  if (metadataProcessingEnabled) {
    // Stabilization
    if (statusMetadata.ok()) {
      stabilizationAlgorithm.addMeasures(metadata.imu);
    }

    videoPipe->resetRotation();
    double yaw = 0, pitch = 0, roll = 0.;

    if (stabilizationEnabled) {
      Quaternion<double> currentOrientation = stabilizationAlgorithm.computeOrientation(date);
      currentOrientation.conjugate().toEuler(yaw, pitch, roll);
      videoPipe->applyRotation(yaw * 180. / M_PI, pitch * 180. / M_PI, roll * 180. / M_PI);
      audioPipe->applyRotation(yaw, pitch, roll);
    }
    qUserOrientation.toEuler(yaw, pitch, roll);
    videoPipe->applyRotation(yaw * 180. / M_PI, pitch * 180. / M_PI, roll * 180. / M_PI);

    // Exposure
    if (statusMetadata.ok()) {
      // update exposure curves
      std::unique_ptr<PanoDefinition> panoWithMetadataExposure =
          exposureProcessor.createUpdatedPano(metadata, getPano(), readerController->getFrameRate(),
                                              readerController->getFrameRate().timestampToFrame(date));
      if (panoWithMetadataExposure) {
        resetPano(*panoWithMetadataExposure.release());
      }
    }
  }

  videoPipeStatus =
      videoPipe->stitchAndExtract(date, readerController->getFrameRate(), inputBuffers, output, extracts, algo);
  readerController->releaseBuffer(inputBuffers);

  return ControllerStatus::fromError(videoPipeStatus);
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::applyRotation(double yaw, double pitch, double roll) {
  videoPipe->applyRotation(yaw, pitch, roll);
  audioPipe->applyRotation(yaw, pitch, roll);
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::resetRotation() {
  videoPipe->resetRotation();
  audioPipe->resetRotation();
}

template <typename VideoPipeline>
Quaternion<double> ControllerImpl<VideoPipeline>::getRotation() const {
  return videoPipe->getRotation();
}

template <typename VideoPipeline>
void ControllerImpl<VideoPipeline>::setSphereScale(const double sphereScale) {
  pano->setSphereScale(sphereScale);
}

// ------------------ explicit instantiations --------------

template class ControllerImpl<PanoPipeline>;
template class ControllerImpl<StereoPipeline>;
}  // namespace Core
}  // namespace VideoStitch
