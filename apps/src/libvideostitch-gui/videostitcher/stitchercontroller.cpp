// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "stitchercontroller.hpp"

#include "stitchercontrollerprogressreporter.hpp"
#include "videostitcher.hpp"

#include "libvideostitch-gui/base/ptvMerger.hpp"
#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"
#include "libvideostitch-gui/mainwindow/LibLogHelpers.hpp"
#include "libvideostitch-gui/utils/gpuhelper.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/utils/stereooutputenum.hpp"
#include "libvideostitch-gui/utils/sourcewidgetlayoututil.hpp"

#include "libvideostitch-base/file.hpp"
#include "libvideostitch-base/yprsignalcaps.hpp"

#include "libvideostitch/context.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/opengl.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/preprocessor.hpp"
#include "libvideostitch/postprocessor.hpp"
#include "libvideostitch/ptv.hpp"

#include <libgpudiscovery/genericDeviceInfo.hpp>

#include <QDesktopWidget>
#include <QApplication>
#include <QRect>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QFileInfo>
#include <QDir>

#ifdef Q_OS_WIN
#include <codecvt>
#endif
#include <iomanip>
#include <memory>

static const unsigned int INPUT_DIGITS(2);

QRect getAvailableScreenSize() {
  return QApplication::desktop()->availableGeometry(QApplication::desktop()->primaryScreen());
}

bool StitcherController::isPlaying() const {
  std::lock_guard<std::mutex> locker(playingMutex);
  return playing;
}

void StitcherController::play() {
  playingMutex.lock();
  playing = true;
  playingMutex.unlock();
  requireNextFrame();
}

bool StitcherController::pause() {
  std::lock_guard<std::mutex> locker(playingMutex);
  bool r = playing;
  playing = false;
  return r;
}

StitcherController::NextFrameAction StitcherController::setNextFrameAction(StitcherController::NextFrameAction action) {
  // First disconnect to be sure to remove old connections
  disconnect(this, &StitcherController::reqNext, this, nullptr);

  playingMutex.lock();
  auto oldFrameAction = nextFrameAction;
  nextFrameAction = action;
  switch (nextFrameAction) {
    case NextFrameAction::None:
      connect(
          this, &StitcherController::reqNext, this, [this]() { stitchRepeatCompressor->pop(); }, Qt::QueuedConnection);
      break;
    case NextFrameAction::Extract:
      connect(this, &StitcherController::reqNext, this, &StitcherController::extractRepeat, Qt::QueuedConnection);
      break;
    case NextFrameAction::Stitch:
      connect(this, &StitcherController::reqNext, this, &StitcherController::stitchRepeat, Qt::QueuedConnection);
      break;
    case NextFrameAction::StitchAndExtract:
      connect(this, &StitcherController::reqNext, this, &StitcherController::stitchAndExtractRepeat,
              Qt::QueuedConnection);
      break;
  }
  playingMutex.unlock();

  // flush out old stitch commands, enqueue one with the new action immediately
  requireNextFrame();
  return oldFrameAction;
}

void StitcherController::requireNextFrame() {
  std::lock_guard<std::mutex> locker(playingMutex);
  if (playing) {
    emit reqNext(stitchRepeatCompressor->add());
  }
}

StitcherController::StitcherController(Controller::DeviceDefinition& device)
    : stitcher(nullptr),
      stitchOutput(nullptr),
      audioPlayer(new AudioPlayer),
      device(device),
      controller(nullptr),
      offscreen(new QOffscreenSurface()),
      logErrStrm(new std::ostream(VideoStitch::Helper::LogManager::getInstance()->getErrorLog())),
      logWarnStrm(new std::ostream(VideoStitch::Helper::LogManager::getInstance()->getWarningLog())),
      logInfoStrm(new std::ostream(VideoStitch::Helper::LogManager::getInstance()->getInfoLog())),
      logVerbStrm(new std::ostream(VideoStitch::Helper::LogManager::getInstance()->getVerboseLog())),
      logDebugStrm(new std::ostream(VideoStitch::Helper::LogManager::getInstance()->getDebugLog())),
      backendInitializerProgressReporter(new BackendInitializerProgressReporter(this)),
      stitchRepeatCompressor(SignalCompressionCaps::createOwned()),
      closing(false),
      actionDone(false),
      stateMachine(this),
      projectOpening(false),
      playing(false),
      nextFrameAction(NextFrameAction::None),
      idle(new QState(&stateMachine)),
      loading(new QState(&stateMachine)),
      loaded(new QState(&stateMachine)) {
  algoOutput.algoOutput = nullptr;
  initializeStitcherLoggers();
  idle->addTransition(this, SIGNAL(projectLoading()), loading);
  loading->addTransition(this, SIGNAL(projectLoaded()), loaded);
  loading->addTransition(this, SIGNAL(cancelProjectLoading()), idle);
  loaded->addTransition(this, SIGNAL(projectClosed()), idle);

  // connect to state actions
  connect(idle, &QState::entered, this, &StitcherController::onProjectClosed);
  connect(loading, &QState::entered, this, &StitcherController::onProjectLoading);
  connect(loaded, &QState::entered, this, &StitcherController::onProjectLoaded);

  stateMachine.setInitialState(idle);
  stateMachine.start();

  // Create an OpenGL context for allocating the surfaces
  // The controller is the parent, to make it reside in the controller's thread
  interopCtx = new QOpenGLContext(this);
  interopCtx->setShareContext(QOpenGLContext::globalShareContext());
  bool oglcr = interopCtx->create();
  Q_ASSERT(interopCtx->shareContext());
  Q_ASSERT(oglcr);
  offscreen->setObjectName("Stitcher controller offscreen surface");
  offscreen->create();
}

void StitcherController::onProjectLoading() {
  projectOpening = true;
  emit reqDisableWindow();
}

void StitcherController::onProjectLoaded() {
  projectOpening = false;
  emit reqEnableWindow();
}

void StitcherController::onProjectClosed() {
  projectOpening = false;
  emit reqEnableWindow();
}

StitcherController::~StitcherController() {
  delete backendInitializerProgressReporter;
  closeProject();
  deleteStitcherLoggers();
  interopCtx->deleteLater();
  offscreen->deleteLater();
}

VideoStitch::FrameRate StitcherController::getFrameRate() const {
  setupLock.lockForRead();
  if (controller) {
    auto frameRate = controller->getFrameRate();
    setupLock.unlock();
    return frameRate;
  } else {
    setupLock.unlock();
    return {100, 1};
  }
}

bool StitcherController::hasInputAudio() const {
  setupLock.lockForRead();
  if (controller) {
    bool hasInputAudio = controller->hasAudio();
    setupLock.unlock();
    return hasInputAudio;
  } else {
    setupLock.unlock();
    return false;
  }
}

bool StitcherController::hasVuMeter(const QString& audioInputId) const {
  setupLock.lockForRead();
  if (controller) {
    const bool hasVuMeter = controller->hasVuMeter(audioInputId.toStdString());
    setupLock.unlock();
    return hasVuMeter;
  } else {
    setupLock.unlock();
    return false;
  }
}

std::vector<double> StitcherController::getRMSValues(const QString& audioInputId) const {
  setupLock.lockForRead();
  if (controller) {
    auto RMSValues = controller->getRMSValues(audioInputId.toStdString());
    setupLock.unlock();
    return RMSValues;
  } else {
    setupLock.unlock();
    return {};
  }
}

std::vector<double> StitcherController::getPeakValues(const QString& audioInputId) const {
  setupLock.lockForRead();
  if (controller) {
    auto peakValues = controller->getPeakValues(audioInputId.toStdString());
    setupLock.unlock();
    return peakValues;
  } else {
    setupLock.unlock();
    return {};
  }
}

void StitcherController::initializeStitcherLoggers() {
  VideoStitch::Logger::setLogStream(VideoStitch::Logger::Debug, logDebugStrm);
  VideoStitch::Logger::setLogStream(VideoStitch::Logger::Verbose, logVerbStrm);
  VideoStitch::Logger::setLogStream(VideoStitch::Logger::Info, logInfoStrm);
  VideoStitch::Logger::setLogStream(VideoStitch::Logger::Warning, logWarnStrm);
  VideoStitch::Logger::setLogStream(VideoStitch::Logger::Error, logErrStrm);
}

void StitcherController::deleteStitcherLoggers() {
  VideoStitch::Logger::setDefaultStreams();

  delete logDebugStrm;
  delete logVerbStrm;
  delete logInfoStrm;
  delete logWarnStrm;
  delete logErrStrm;
}

void StitcherController::delayedUpdate() {
  QTimer::singleShot(1000, this, SIGNAL(reqUpdate()));  // VSA-5054 & VSA-4923: we need to delay the update
}

VideoStitch::Potential<VideoStitch::Core::PanoDefinition> StitcherController::createAPanoTemplateFromCalibration(
    QString calibrationFile, QString& errorString) const {
  VideoStitch::Logger::get(VideoStitch::Logger::Info)
      << "Trying to apply calibration file: " << calibrationFile.toStdString() << std::endl;

  // Check the file extension first
  if (File::getTypeFromFile(calibrationFile) != File::CALIBRATION) {
    errorString = StitcherController::tr("Template couldn't be applied: invalid file extension");
    return nullptr;
  }

  // Convert it from pts/pto
  VideoStitch::Potential<VideoStitch::Core::PanoDefinition> tmpDefinition(
      VideoStitch::Core::PanoDefinition::parseFromPto(calibrationFile.toStdString(),
                                                      getProjectPtr()->getPanoConst().get()));

  if (!tmpDefinition.ok()) {
    return VideoStitch::Status(VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::InvalidConfiguration,
                               StitcherController::tr("Could not apply calibration template").toStdString(),
                               tmpDefinition.status());
  }

  // Keep the inputs reader_config and frame_offsets
  std::unique_ptr<VideoStitch::Core::PanoDefinition> templatePanoDef(tmpDefinition.release());
  if (templatePanoDef->numInputs() != getProjectPtr()->getPanoConst()->numInputs()) {
    return VideoStitch::Status(
        VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::InvalidConfiguration,
        StitcherController::tr("Could not apply calibration template: number of inputs not matching").toStdString());
  }

  // Replace the readers
  for (readerid_t index = 0; index < templatePanoDef->numInputs(); ++index) {
    templatePanoDef->getInput(index).setReaderConfig(
        getProjectPtr()->getPanoConst()->getInput(index).getReaderConfig().clone());
    templatePanoDef->getInput(index).setFrameOffset(getProjectPtr()->getPanoConst()->getInput(index).getFrameOffset());
  }
  templatePanoDef->setHeight(getProjectPtr()->getPanoConst()->getHeight());
  templatePanoDef->setWidth(getProjectPtr()->getPanoConst()->getWidth());

  return templatePanoDef.release();
}

VideoStitch::Potential<VideoStitch::Core::PanoDefinition> StitcherController::createAPanoTemplateFromProject(
    QString projectFile) const {
  Q_ASSERT(File::getTypeFromFile(projectFile) == File::PTV || File::getTypeFromFile(projectFile) == File::VAH);

  VideoStitch::Potential<VideoStitch::Ptv::Parser> parser(VideoStitch::Ptv::Parser::create());
  if (!parser.ok()) {
    return VideoStitch::Status(VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::SetupFailure,
                               StitcherController::tr("Could not initialize the project file parser").toStdString(),
                               parser.status());
  }

  if (!parser->parse(projectFile.toStdString())) {
    return VideoStitch::Status(VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::InvalidConfiguration,
                               parser->getErrorMessage());
  }

  std::unique_ptr<VideoStitch::Ptv::Value> templatePanoConfig(parser->getRoot().has("pano")->clone());
  std::unique_ptr<VideoStitch::Core::PanoDefinition> templateDefinition(
      VideoStitch::Core::PanoDefinition::create(*templatePanoConfig));
  std::stringstream errorStream;

  // Let's check that the pano definition is valid
  if (!templateDefinition || !templateDefinition->validate(errorStream)) {
    return VideoStitch::Status(
        VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::InvalidConfiguration,
        StitcherController::tr("The imported panorama definition is invalid. %0\nPlease check your calibration file.")
            .arg(QString::fromStdString(errorStream.str()))
            .toStdString(),
        parser.status());
  }

  // FIXME : add an empty input for each audio only input in the pano (with the same index)
  std::vector<VideoStitch::Ptv::Value*> templateInputs = templatePanoConfig->get("inputs")->asList();
  videoreaderid_t nbTemplateVideoInput = 0;
  size_t index = 0;
  while (index < templateInputs.size()) {
    VideoStitch::Ptv::Value* input = templateInputs[index];
    if (input->has("video_enabled") && !input->has("video_enabled")->asBool()) {
      delete input;
      templateInputs.erase(templateInputs.begin() + index);
      continue;
    }
    delete input->remove("reader_config");
    delete input->remove("frame_offset");
    delete input->remove("ev");
    delete input->remove("red_corr");
    delete input->remove("blue_corr");
    delete input->remove("green_corr");
    delete input->remove("width");
    delete input->remove("height");
    delete input->remove("group");
    ++nbTemplateVideoInput;
    ++index;
  }
  // Use the cleared template inputs
  templatePanoConfig->get("inputs")->asList() = templateInputs;

  // Check for same ammount of inputs.
  if (nbTemplateVideoInput != getProjectPtr()->getPanoConst()->numVideoInputs()) {
    return VideoStitch::Status(VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::InvalidConfiguration,
                               StitcherController::tr("The number of video inputs of the template doesn't match. "
                                                      "The calibration template has %0 video inputs,"
                                                      "the current panorama has %1")
                                   .arg(nbTemplateVideoInput)
                                   .arg(getProjectPtr()->getNumInputs())
                                   .toStdString(),
                               parser.status());
  }

  // Strip the unwanted values from the PTV/VAH pano config.
  delete templatePanoConfig->remove("global_orientation");
  delete templatePanoConfig->remove("stabilization");
  delete templatePanoConfig->remove("ev");
  delete templatePanoConfig->remove("red_corr");
  delete templatePanoConfig->remove("blue_corr");
  delete templatePanoConfig->remove("green_corr");
  delete templatePanoConfig->remove("height");
  delete templatePanoConfig->remove("width");
  // Strip calibration data
  delete templatePanoConfig->remove("calibration_control_points");
  delete templatePanoConfig->remove("rig");
  delete templatePanoConfig->remove("cameras");

  // Save audio inputs from the project (not the ptv/vah)
  // TODO implement a calibration import not based on PTV /VAH values.
  std::vector<VideoStitch::Ptv::Value*> audioInputs;
  for (audioreaderid_t i = 0; i < getProjectPtr()->getPanoConst()->numAudioInputs(); ++i) {
    const readerid_t index = getProjectPtr()->getPanoConst()->convertAudioInputIndexToInputIndex(i);
    // Audio only inputs
    if (!getProjectPtr()->getPanoConst()->getInput(index).getIsVideoEnabled()) {
      audioInputs.push_back(getProjectPtr()->getPano()->popInput(index)->serialize());
    }
  }

  std::unique_ptr<VideoStitch::Ptv::Value> panoConfig(getProjectPtr()->getPanoConst()->serialize());
  VideoStitch::Helper::PtvMerger::mergeValue(panoConfig.get(), templatePanoConfig.get());

  // Strip the cropping
  delete panoConfig->remove("crop_left");
  delete panoConfig->remove("crop_right");
  delete panoConfig->remove("crop_top");
  delete panoConfig->remove("crop_bottom");

  std::unique_ptr<VideoStitch::Core::PanoDefinition> panoDefinitionNew(
      VideoStitch::Core::PanoDefinition::create(*panoConfig));
  if (!panoDefinitionNew || !panoDefinitionNew->validate(errorStream)) {
    return VideoStitch::Status(VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::InvalidConfiguration,
                               StitcherController::tr("The calibration couldn't be applied. %0")
                                   .arg(QString::fromStdString(errorStream.str()))
                                   .toStdString(),
                               parser.status());
  }

  // Recover the audio inputs
  for (VideoStitch::Ptv::Value* audioInput : audioInputs) {
    readerid_t pos = panoDefinitionNew->numInputs();
    panoDefinitionNew->insertInput(VideoStitch::Core::InputDefinition::create(*audioInput), ++pos);
  }

  return panoDefinitionNew.release();
}

// -------------------------- Stitching -------------------------------------

void StitcherController::stitchOnce() {
  assert(QThread::currentThread() == this->thread());
  if (stitcher == nullptr) {
    return;
  }
  stitcher->stitch();
}

void StitcherController::extractOnce() {
  assert(QThread::currentThread() == this->thread());
  if (stitcher == nullptr) {
    return;
  }
  stitcher->extract();
}

void StitcherController::stitchAndExtractOnce() {
  assert(QThread::currentThread() == this->thread());
  if (stitcher == nullptr) {
    return;
  }
  stitcher->stitchAndExtract();
}

void StitcherController::restitchOnce() {
  assert(QThread::currentThread() == this->thread());
  if (stitcher == nullptr) {
    return;
  }
  stitcher->restitch();
}

void StitcherController::reextractOnce() {
  assert(QThread::currentThread() == this->thread());
  if (stitcher == nullptr) {
    return;
  }
  stitcher->reextract();
}

void StitcherController::restitchAndExtractOnce() {
  assert(QThread::currentThread() == this->thread());
  if (stitcher == nullptr) {
    return;
  }
  stitcher->restitchAndExtract();
}

void StitcherController::stitchRepeat(SignalCompressionCaps* comp) {
  if (comp->pop() > 0) {
    return;
  }
  stitchOnce();
  requireNextFrame();
}

void StitcherController::extractRepeat(SignalCompressionCaps* comp) {
  if (comp->pop() > 0) {
    return;
  }
  extractOnce();
  requireNextFrame();
}

void StitcherController::stitchAndExtractRepeat(SignalCompressionCaps* comp) {
  if (comp->pop() > 0) {
    return;
  }
  stitchAndExtractOnce();
  requireNextFrame();
}

// -------------------------- Open ---------------------------------------------------

void StitcherController::setProjectOpening(const bool b) { projectOpening = b; }

bool StitcherController::isProjectOpening() const { return projectOpening; }

bool StitcherController::openProject(const QString& PTVFile, int customWidth, int customHeight) {
  StitcherControllerProgressReporter progressReporter(this);
  if (stateMachine.configuration().contains(loaded)) {
    closeProject();
  }
  auto cancel = [this]() {
    emit cancelProjectLoading();
    return false;
  };
  emit projectLoading();

  const QFileInfo fInfo = QFileInfo(PTVFile);
  QDir::setCurrent(fInfo.absolutePath());

  // 1- Parse the project to get the configuration
  VideoStitch::Potential<VideoStitch::Ptv::Parser> parser(VideoStitch::Ptv::Parser::create());
  if (!parser.ok()) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        StitcherController::tr("Could not initialize the project file parser."));
    return cancel();
  }
  if (!parser->parse(PTVFile.toStdString())) {
    emit notifyErrorMessage({VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::InvalidConfiguration,
                             StitcherController::tr("Could not parse the project file %0.")
                                 .arg(QString::fromStdString(parser->getErrorMessage()))
                                 .toStdString()},
                            true);
    return cancel();
  }
  progressReporter.setProgress(30);

  // 2- Create the Project
  std::unique_ptr<VideoStitch::Ptv::Value> ptv(parser->getRoot().clone());
  if (!ptv->asObject().has("pano")) {
    emit notifyErrorMessage({VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::InvalidConfiguration,
                             StitcherController::tr("Missing panorama parameters. Aborting.").toStdString()},
                            true);
    return cancel();
  }

  createProject();
  VideoStitch::Ptv::Value* panoConfigUnverified = ptv->asObject().get("pano");
  delete panoConfigUnverified->remove("crop_left");
  delete panoConfigUnverified->remove("crop_right");
  delete panoConfigUnverified->remove("crop_top");
  delete panoConfigUnverified->remove("crop_bottom");
  if (!getProjectPtr()->load(*ptv)) {
    emit notifyErrorMessage({VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::InvalidConfiguration,
                             StitcherController::tr("Incorrect project parameters. Aborting.").toStdString()},
                            true);
    return cancel();
  }
  checkPanoDeprecatedFeatures(*panoConfigUnverified);

  // 3- Application-specific checks on the project
  if (!checkProject()) {
    emit openFromInputFailed();
    return cancel();
  }
  progressReporter.setProgress(50);
  bool success = open(&progressReporter, 0, customWidth, customHeight);
  if (!success) {
    return cancel();
  }
  progressReporter.finishProgress();
  emit notifyProjectOpened();
  emit projectLoaded();
  return true;
  ;
}

bool StitcherController::open(StitcherControllerProgressReporter* progressReporter, int frame, int customWidth,
                              int customHeight) {
  Q_ASSERT(getProjectPtr() != nullptr);
  auto cancel = [this]() {
    closeProject();
    emit reqCleanStitcher();
    return false;
  };
  auto waitForAction = [this](std::unique_lock<std::mutex>& lock, std::condition_variable& condition) {
    condition.wait(lock, [this] { return this->closing || this->actionDone; });
    if (closing) {
      return false;
    }
    return true;
  };
  // This code should only be run by the controller Thread
  Q_ASSERT(QThread::currentThread() == this->thread());

  bool oglmkc = interopCtx->makeCurrent(offscreen);
  Q_ASSERT(oglmkc);

  // Make sure that we are running on a valid device
  VideoStitch::Status status = VideoStitch::GPU::Context::setDefaultBackendDeviceAndCheck(device.device);
  if (!status.ok()) {
    VideoStitch::GPU::showGPUInitializationError(device.device, status.getErrorMessage());
    return cancel();
  }

  // 4- Instantiate the controller
  if ((customHeight != 0) || (customWidth != 0)) {
    getProjectPtr()->updateSize(customWidth, customHeight);
  }

  VideoStitch::Input::ReaderFactory* readerFactory = createReaderFactory();
  Q_ASSERT(getProjectPtr()->getPanoConst().get() != nullptr);
  auto potentialController = makeController(getProjectPtr(), readerFactory);
  if (!potentialController.ok()) {
    setupLock.lockForWrite();
    controller = nullptr;
    setupLock.unlock();
    // Ignore user cancellation
    if (!potentialController.status().hasUnderlyingCause(VideoStitch::ErrType::OperationAbortedByUser)) {
      forwardStitcherError(VideoStitch::Core::ControllerStatus::fromError(
                               {VideoStitch::Origin::Stitcher, VideoStitch::ErrType::SetupFailure,
                                StitcherController::tr("Input Error").toStdString(), potentialController.status()}),
                           true);
    }
    emit openFromInputFailed();
    return cancel();
  }
  setupLock.lockForWrite();
  controller = potentialController.release();
  setupLock.unlock();
  if (progressReporter) {
    progressReporter->setProgress(70);
  }

  // 5- Instantiate the callbacks updating the source thumbnails widgets
  std::vector<std::vector<std::shared_ptr<VideoStitch::Core::SourceSurface>>> allSurfs;
  extractsOutputs.clear();
  const QList<QString> urlList = getProjectPtr()->getInputNames();
  std::vector<std::tuple<VideoStitch::Input::VideoReader::Spec, bool, std::string>> inputs;

  for (int id = 0; id < (int)getProjectPtr()->getPanoConst()->numInputs(); ++id) {
    const VideoStitch::Core::InputDefinition& inputDef = getProjectPtr()->getPanoConst()->getInput(id);
    if (inputDef.getIsVideoEnabled()) {
      VideoStitch::Input::VideoReader::Spec spec = controller->getReaderSpec(id);
      auto potSurf = VideoStitch::Core::OpenGLAllocator::createSourceSurface(spec.width, spec.height);
      if (!potSurf.ok()) {
        forwardStitcherError(VideoStitch::Core::ControllerStatus::fromError(potSurf.status()), false);
        return cancel();
      }
      potSurf.object()->sourceId = id;
      allSurfs.push_back({std::shared_ptr<VideoStitch::Core::SourceSurface>(potSurf.release())});
      inputs.emplace_back(std::make_tuple(spec, true, urlList.at(id).toStdString()));
    }
  }

  std::vector<std::shared_ptr<VideoStitch::Core::SourceRenderer>> sourceRenderers;
  {
    std::unique_lock<std::mutex> lk(conditionMutex);
    actionDone = false;
    emit reqCreateThumbnails(inputs, &sourceRenderers);
    if (!waitForAction(lk, openCondition)) {
      return cancel();
    }
  }
  int i = 0;
  for (int id = 0; id < (int)getProjectPtr()->getPanoConst()->numInputs(); ++id) {
    const VideoStitch::Core::InputDefinition& inputDef = getProjectPtr()->getPanoConst()->getInput(id);
    if (inputDef.getIsVideoEnabled()) {
      extractsOutputs[id] =
          controller->createAsyncExtractOutput(id, allSurfs[i], sourceRenderers[i], nullptr).release();
      ++i;
    }
  }

  // 6- Instantiate the callbacks updating the panorama widget and the Oculus widget
  std::vector<std::shared_ptr<typename Controller::Output::Writer>> cbks;
  std::vector<std::shared_ptr<VideoStitch::Core::PanoSurface>> surfs;
  std::vector<std::shared_ptr<VideoStitch::Core::PanoRenderer>> panoRenderers;
  for (int i = 0; i < 2; ++i) {
    if (getProjectPtr()->getPanoConst()->getProjection() == VideoStitch::Core::PanoProjection::Equirectangular) {
      VideoStitch::Potential<VideoStitch::Core::PanoOpenGLSurface> potSurf =
          VideoStitch::Core::OpenGLAllocator::createPanoSurface(getProjectPtr()->getPanoConst()->getWidth(),
                                                                getProjectPtr()->getPanoConst()->getHeight());
      if (!potSurf.ok()) {
        forwardStitcherError(VideoStitch::Core::ControllerStatus::fromError(potSurf.status()), false);
        return cancel();
      }
      surfs.push_back(std::shared_ptr<VideoStitch::Core::PanoSurface>(potSurf.release()));
    } else if (getProjectPtr()->getPanoConst()->getProjection() == VideoStitch::Core::PanoProjection::Cubemap ||
               getProjectPtr()->getPanoConst()->getProjection() ==
                   VideoStitch::Core::PanoProjection::EquiangularCubemap) {
      VideoStitch::Potential<VideoStitch::Core::CubemapOpenGLSurface> potSurf =
          VideoStitch::Core::OpenGLAllocator::createCubemapSurface(
              getProjectPtr()->getPanoConst()->getLength(), getProjectPtr()->getPanoConst()->getProjection() ==
                                                                VideoStitch::Core::PanoProjection::EquiangularCubemap);
      if (!potSurf.ok()) {
        forwardStitcherError(VideoStitch::Core::ControllerStatus::fromError(potSurf.status()), false);
        return cancel();
      }
      surfs.push_back(std::shared_ptr<VideoStitch::Core::PanoSurface>(potSurf.release()));
    } else {
      forwardStitcherError(
          VideoStitch::Status{VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::InvalidConfiguration,
                              "Unknown projection for this panorama."},
          false);
      return cancel();
    }
  }
  {
    std::unique_lock<std::mutex> lk(conditionMutex);
    actionDone = false;
    emit reqCreatePanoView(&panoRenderers);
    if (!waitForAction(lk, openCondition)) {
      return cancel();
    }
  }
  panoRenderers.push_back(audioPlayer);
  VideoStitch::Potential<typename Controller::Output> potStitchOutput =
      controller->createAsyncStitchOutput(surfs, panoRenderers, cbks);
  if (!potStitchOutput.ok()) {
    forwardStitcherError(VideoStitch::Core::ControllerStatus::fromError(potStitchOutput.status()), false);
    return cancel();
  }
  controller->addAudioOutput(audioPlayer);
  stitchOutput = potStitchOutput.release();

  // 7- Instantiate the stitchers
  {
    std::vector<VideoStitch::Core::ExtractOutput*> flatOutputs;
    for (auto kv : extractsOutputs) {
      flatOutputs.push_back(kv.second);
    }
    stitcher = new VideoStitcher<Controller>(this, device, *getProjectPtr(), *controller, *stitchOutput, flatOutputs,
                                             algoOutput, setupLock);
    connect(stitcher, &VideoStitcher<Controller>::snapshotPanoramaExported, this,
            &StitcherController::snapshotPanoramaExported);
    connect(stitcher, &VideoStitcher<Controller>::notifyErrorMessage, this, &StitcherController::forwardStitcherError);

    VideoStitch::Status initStatus = stitcher->init();

    if (!initStatus.ok()) {
      forwardStitcherError(VideoStitch::Core::ControllerStatus::fromError(initStatus), false);
      return cancel();
    }
  }
  if (progressReporter) {
    progressReporter->setProgress(90);
  }

  emit statusMsg(StitcherController::tr("Created a %0 x %1 projection ('%2')")
                     .arg(QString::number(getProjectPtr()->getPanoConst()->getWidth()),
                          QString::number(getProjectPtr()->getPanoConst()->getHeight()),
                          getProjectPtr()->getProjection()));

  // 8- When reloading seek to previous frame
  if (frame != 0) {
    controller->seekFrame(frame);
  }

  // 9- Announce myself
  emit projectInitialized(getProjectPtr());

  // 10- Do something specific to the application!
  finishProjectOpening();

  // 11- Stitch the initial frame!
  stitchAndExtractOnce();
  return true;
}

void StitcherController::closingProject() {
  // This code is intended to be run only by the main Thread
  Q_ASSERT(QThread::currentThread() == QApplication::instance()->thread());

  conditionMutex.lock();
  closing = true;
  conditionMutex.unlock();
  openCondition.notify_all();

  preCloseProject();
}

bool StitcherController::unregisterInputExtractor(int inputId, const QString& name) {
  if (extractsOutputs.find(inputId) != extractsOutputs.end()) {
    return extractsOutputs[inputId]->removeWriter(name.toStdString());
  } else {
    return false;
  }
}

void StitcherController::updateAfterModifyingPrePostProcessor() {
  // force reading the input frames again
  assert(QThread::currentThread() == this->thread());
  controller->seekFrame(controller->getCurrentFrame());
  stitchAndExtractOnce();
  delayedUpdate();
}

void StitcherController::checkPanoDeprecatedFeatures(const VideoStitch::Ptv::Value& pano) {
  const std::vector<VideoStitch::Ptv::Value*>& inputs = pano.has("inputs")->asList();
  for (const VideoStitch::Ptv::Value* input : inputs) {
    if (input->has("viewpoint_model") && input->has("viewpoint_model")->asString() == "ptgui") {
      QString message = StitcherController::tr(
                            "The support for PTGui viewpoint correction has been deprecated, you should re-calibrate "
                            "the project with %0 internal calibration")
                            .arg(QCoreApplication::applicationName());
      MsgBoxHandler::getInstance()->generic(message, StitcherController::tr("Warning"), WARNING_ICON);
      return;
    }
  }
}

void StitcherController::forwardBackendProgress(const QString& message, double progress) {
  emit notifyBackendCompileProgress(message, progress);
}

void StitcherController::tryCancelBackendCompile() {
  Q_ASSERT(backendInitializerProgressReporter);
  backendInitializerProgressReporter->tryToCancel();
}

void StitcherController::onActivateAudioPlayback(const bool b) { audioPlayer->onActivatePlayBack(b); }

// -------------------------- Reconfigure -------------------------------------------

void StitcherController::onReset(SignalCompressionCaps* signalCompressor) {
  if (signalCompressor && signalCompressor->pop() > 0) {
    return;
  }
  reset();
  emit notifyStitcherReset();
}

void StitcherController::onResetRig(SignalCompressionCaps* signalCompressor) {
  if (signalCompressor && signalCompressor->pop() > 0) {
    return;
  }
  onResetRig();
}

void StitcherController::changePano(VideoStitch::Core::PanoDefinition* panoDef) {
  getProjectPtr()->setPano(panoDef);
  reset();
}

void StitcherController::clearCalibration() { getProjectPtr()->getPano()->resetCalibration(); }

// -------------------------- Crop -------------------------------------------

void StitcherController::applyCrops(const QVector<Crop>& crops, const InputLensClass::LensType lensType) {
  Q_ASSERT_X(crops.size() == getProjectPtr()->getPanoConst()->numVideoInputs(), "StitcherController",
             "Crops inputs and input numbers are different");
  for (auto input = 0; input < crops.size(); ++input) {
    getProjectPtr()->setInputCrop(input, crops.at(input), lensType, false);
  }
}

// -------------------------- Close -------------------------------------------

void StitcherController::closeProject() {
  // block until stitchers are deleted, otherwise we can't deleteController
  if (stitcher) {
    stitcher->closeProject();
    stitcher->deleteLater();
    stitcher = nullptr;
  }
  setupLock.lockForWrite();
  if (controller) {
    delete stitchOutput;
    for (int i = 0; i < (int)extractsOutputs.size(); ++i) {
      delete extractsOutputs[i];
    }
    stitchOutput = nullptr;
    deleteController(controller);
    controller = nullptr;
  }
  setupLock.unlock();

  extractsOutputs.clear();
  emit stitcherClosed();
  emit projectClosed();
}

// -------------------------- Save --------------------------------------------

bool StitcherController::saveProject(const QString& outFile, const VideoStitch::Ptv::Value* thisProjectCopy) {
  if (!getProjectPtr()->isInit()) {
    return false;
  }
  auto mode = (outFile.endsWith(".ptvb")) ? std::ios_base::out | std::ios_base::binary : std::ios_base::out;
  std::ofstream ofs;
  std::string filename = outFile.toStdString();
#ifdef Q_OS_WIN
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  std::wstring wideFilename = converter.from_bytes(filename);
  ofs.open(wideFilename, mode);
#else
  ofs.open(filename, mode);
#endif
  if (!ofs.is_open()) {
    emit statusMsg("Couldn't open file " + outFile + " for save.");
    return false;
  }

  ofs.clear();

  // XXX TODO FIXME beware of save when in interactive tab
  const VideoStitch::Ptv::Value* root = thisProjectCopy;

  // if a serialized project was not provided, serialize the current project
  if (thisProjectCopy == nullptr) {
    root = getProjectPtr()->serialize();
  }

  if (outFile.endsWith(".ptvb")) {
    root->printUBJson(ofs);
  } else {
    root->printJson(ofs);
  }

  // delete the serialized project if it was not passed to this function
  if (thisProjectCopy == nullptr) {
    delete root;
    getProjectPtr()->markAsSaved();
  }
  getProjectPtr()->updateFileFormat();
  return true;
}

// -------------------------- Snapshots -------------------------------

void StitcherController::onSnapshotPanorama(const QString& filename) {
  assert(QThread::currentThread() == this->thread());
  stitcher->onSnapshotPanorama(filename);
}

QStringList StitcherController::onSnapshotSources(const QString& directory) {
  assert(QThread::currentThread() == this->thread());
  QStringList outputFiles;
  std::vector<VideoStitch::Ptv::Value*> outputConfigs;
  for (int id = 0; id < (int)getProjectPtr()->getPanoConst()->numInputs(); ++id) {
    const VideoStitch::Core::InputDefinition& inputDef = getProjectPtr()->getPanoConst()->getInput(id);
    if (inputDef.getIsVideoEnabled()) {
      VideoStitch::Ptv::Value* outputConfig = VideoStitch::Ptv::Value::emptyObject();
      outputConfig->get("type")->asString() = "jpg";
      std::stringstream file;
      file << "input-" << std::setfill('0') << std::setw(INPUT_DIGITS) << id;
      std::string outFn = (directory + QDir::separator() + QString::fromStdString(file.str())).toStdString();
      file << ".jpg";
      outputFiles << QString::fromStdString(file.str());
      outputConfig->get("filename")->asString() = outFn;
      outputConfig->get("numbered_digits")->asInt() = 0;
      outputConfig->get("width")->asInt() = inputDef.getWidth();
      outputConfig->get("height")->asInt() = inputDef.getHeight();
      outputConfigs.push_back(outputConfig);
    }
  }

  stitcher->onSnapshotSources(outputConfigs);

  // return the absolute snapshots paths
  QStringList snapshots;
  foreach (QString img, outputFiles) { snapshots << directory + QString(QDir::separator()) + img; }
  return snapshots;
}

// -------------------------- Orientation ------------------------------------

void StitcherController::rotatePanorama(YPRSignalCaps* rotations, bool rsttch) {
  double yaw, pitch, roll;
  rotations->popAll(yaw, pitch, roll);
  if (yaw != 0.0 || pitch != 0.0 || roll != 0.0) {
    stitcher->rotatePanorama(yaw, pitch, roll);
    if (rsttch) {
      restitchOnce();
    }
  }
}

// -------------------------- Sphere Scale ----------------------------------------

void StitcherController::setSphereScale(const double sphereScale, bool rsttch) {
  if (sphereScale > 0.) {
    setupLock.lockForWrite();
    if (controller) {
      controller->setSphereScale(sphereScale);
      setupLock.unlock();
      if (rsttch) {
        restitchOnce();
      }
    } else {
      setupLock.unlock();
    }
  }
}

// -------------------------- Processors ----------------------------------------

void StitcherController::toggleInputNumbers(bool draw) {
  if (!getProjectPtr()->isInit()) {
    Q_ASSERT(0);
    return;
  }

  for (int i = 0; i < int(getProjectPtr()->getNumInputs()); ++i) {
    if (draw) {
      // skip non-video inputs
      if (!getProjectPtr()->getPanoConst()->getInput(i).getIsVideoEnabled()) {
        continue;
      }
      std::unique_ptr<VideoStitch::Ptv::Value> value(VideoStitch::Ptv::Value::emptyObject());
      std::stringstream val;
      val << i;
      value->get("type")->asString() = "expr";
      value->get("value")->asString() = val.str();
      value->get("color")->asString() = ORAH_COLOR;
      VideoStitch::Potential<VideoStitch::Core::PreProcessor> p = VideoStitch::Core::PreProcessor::create(*value);
      if (!p.ok()) {
        VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
            StitcherController::tr("Could not initialize the preprocessor."));
      } else {
        setupLock.lockForWrite();
        controller->setPreProcessor(i, p.release());
        setupLock.unlock();
      }
    } else {
      setupLock.lockForWrite();
      controller->setPreProcessor(i, nullptr);
      setupLock.unlock();
    }
  }
  updateAfterModifyingPrePostProcessor();
  emit inputNumbersToggled(draw);
}

void StitcherController::setAudioDelay(int delay) {
  setupLock.lockForWrite();
  if (controller) {
    controller->setAudioDelay((double)delay);
  }
  setupLock.unlock();
}

void StitcherController::setAudioInput(const QString& name) {
  setupLock.lockForWrite();
  controller->setAudioInput(name.toStdString());
  setupLock.unlock();
}

void StitcherController::registerSourceRender(std::shared_ptr<VideoStitch::Core::SourceRenderer> renderer,
                                              const int inputId) {
  if (!extractsOutputs[inputId]->addRenderer(renderer)) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(StitcherController::tr("Could not start renderer"));
  }
}

void StitcherController::unregisterSourceRender(const QString name, const int inputId) {
  if (!extractsOutputs[inputId]->removeRenderer(name.toStdString())) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        StitcherController::tr("Could not unregister renderer"));
  }
}

void StitcherController::forwardStitcherError(const VideoStitch::Core::ControllerStatus status, bool needToExit) {
  switch (status.getCode()) {
    case VideoStitch::Core::ControllerStatusCode::EndOfStream:
      emit notifyEndOfStream();
      return;
    case VideoStitch::Core::ControllerStatusCode::ErrorWithStatus:
      if (status.getStatus().hasUnderlyingCause(VideoStitch::ErrType::OutOfResources) && getProjectPtr() != nullptr) {
        emit projectInitialized(getProjectPtr());
        emit reqResetDimensions(getProjectPtr()->getPanoConst()->getWidth(),
                                getProjectPtr()->getPanoConst()->getHeight(), getProjectPtr()->getInputNames());
      } else {
        emit notifyErrorMessage(status.getStatus(), needToExit);
      }
      return;
    case VideoStitch::Core::ControllerStatusCode::Ok:
    default:
      // shouldn't be called in any other case
      assert(false);
      return;
  }
}

std::vector<int> StitcherController::listGpus(const std::vector<VideoStitch::Core::PanoDeviceDefinition>& devDef) {
  std::vector<int> ids;
  std::transform(devDef.begin(), devDef.end(), std::back_inserter(ids),
                 [](VideoStitch::Core::PanoDeviceDefinition d) { return d.device; });
  return ids;
}

VideoStitch::Core::PotentialController StitcherController::makeController(
    ProjectDefinition* proj, VideoStitch::Input::ReaderFactory* readerFactory) {
  // initialize backend
  Q_ASSERT(backendInitializerProgressReporter);
  backendInitializerProgressReporter->reset();
  auto backendStatus = VideoStitch::GPU::Context::compileAllKernelsOnSelectedDevice(device.device, true,
                                                                                    backendInitializerProgressReporter);
  emit notifyBackendCompileDone();
  if (!backendStatus.ok()) {
    if (!backendStatus.hasUnderlyingCause(VideoStitch::ErrType::OperationAbortedByUser)) {
      VideoStitch::GPU::showGPUInitializationError(device.device, backendStatus.getErrorMessage());
    }
    return backendStatus;
  }

  return createController(*proj->getPanoConst().get(), *proj->getImageMergerFactory().get(),
                          *proj->getImageWarperFactory().get(), *proj->getImageFlowFactory().get(), readerFactory,
                          *proj->getAudioPipeConst().get());
}
