// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "projectworkwidget.hpp"

#include "cropinputcontroller.hpp"
#include "guiconstants.hpp"
#include "outputtabwidget.hpp"
#include "outputscontroller.hpp"
#include "widgetsmanager.hpp"
#include "plugin/outputplugincontroller.hpp"
#include "plugin/inputplugincontroller.hpp"
#include "calibration/calibrationactioncontroller.hpp"
#include "calibration/calibrationupdatecontroller.hpp"
#include "configurations/configexposurewidget.hpp"
#include "configurations/configpanorama.hpp"
#include "generic/genericdialog.hpp"
#include "videostitcher/globallivecontroller.hpp"
#include "videostitcher/liveoutputfactory.hpp"
#include "videostitcher/liveoutputlist.hpp"

#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"
#include "libvideostitch-gui/mainwindow/statemanager.hpp"
#include "libvideostitch-gui/mainwindow/timeconverter.hpp"
#include "libvideostitch-gui/mainwindow/ui_header/progressreporterwrapper.hpp"
#include "libvideostitch-gui/videostitcher/audioplayer.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"
#include "libvideostitch-gui/widgets/deviceinteractivewidget.hpp"

#include "libvideostitch-gui/mainwindow/LibLogHelpers.hpp"

#include "libvideostitch-base/logmanager.hpp"
#include "libvideostitch-base/yprsignalcaps.hpp"

#include "libvideostitch/opengl.hpp"
#include "libvideostitch/plugin.hpp"
#include "libvideostitch/ptv.hpp"

#include <QLabel>
#include <QList>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QUrl>

#include <vector>

int getOpenGLDevice() {
  QOpenGLContext ctx;
  ctx.create();
  QOffscreenSurface surface;
  surface.create();
  ctx.makeCurrent(&surface);
  std::vector<int> glDevices = VideoStitch::getGLDevices();
  ctx.doneCurrent();
  return glDevices[0];
}

ProjectWorkWidget::ProjectWorkWidget(QWidget* const parent)
    : QWidget(parent),
      projectDefinition(nullptr),
      outputsController(nullptr),
      calibrationActionController(nullptr),
      calibrationUpdateController(nullptr),
      exposureController(nullptr),
      inputPluginController(nullptr),
      outputPluginController(nullptr),
      cropInputController(nullptr),
      nextFrameAction(StitcherController::NextFrameAction::None) {
  setupUi(this);
  stackedWidget->setCurrentWidget(pageProjectTabs);
  StateManager::getInstance()->registerObject(this);
}

ProjectWorkWidget::~ProjectWorkWidget() {
  if (projectTabs->sourcesTabWidget) {
    projectTabs->sourcesTabWidget->getSourcesWidget()->clearThumbnails();
  }
  emit reqThreadQuit();

  delete outputsController;
  delete calibrationActionController;
  delete calibrationUpdateController;
  delete exposureController;
  delete inputPluginController;
  delete outputPluginController;
  StitcherController* videoStitcher = GlobalController::getInstance().getController();
  if (videoStitcher) {
    videoStitcher->closingProject();
  }
  GlobalLiveController::getInstance().deleteController();

  stitcherControllerThread.wait();
  gpuInfoUpdater.wait();
}

void ProjectWorkWidget::startApplication() {
  gpuInfoUpdater.start();
  emit reqForceGPUInfoRefresh();
}

void ProjectWorkWidget::openFile(QVector<int> devices, const QFileInfo& fileName, const int customWidth,
                                 const int customHeight) {
  createStitcherController(devices);
  emit reqChangeState(GUIStateCaps::disabled);
  informationBar->setCurrentProjectName(fileName.baseName());
  ProjectFileHandler::getInstance()->setFilename(fileName.absoluteFilePath());
  GlobalController::getInstance().getController()->setProjectOpening(true);
  emit reqOpenVAHFile(fileName.absoluteFilePath(), customWidth, customHeight);
}

void ProjectWorkWidget::startNewProject(QVector<int> devices, const QString& name) {
  createStitcherController(devices);
  emit reqChangeState(GUIStateCaps::disabled);
  informationBar->setCurrentProjectName(name);
  const QString& filePath = QDir::fromNativeSeparators(getProjectsPath() + QDir::separator() + name + VAH_FILE);
  ProjectFileHandler::getInstance()->setFilename(filePath);
  emit reqCreateNewProject();
  emit reqChangeState(GUIStateCaps::idle);
}

void ProjectWorkWidget::saveProject() { emit reqSave(ProjectFileHandler::getInstance()->getFilename()); }

void ProjectWorkWidget::initializeMainTab() {
  // Create tabs and controllers
  projectTabs->addAllTabs(GlobalLiveController::getInstance().getController(),
                          GlobalController::getInstance().getController());
  updateClockForTab(projectTabs->currentIndex());

  // lazy controllers initialization
  if (!outputsController) {
    outputsController =
        new OutputsController(informationBar, projectTabs->outPutTabWidget, projectTabs->configurationTabWidget);
  }
  if (!calibrationActionController) {
    calibrationActionController = new CalibrationActionController(projectTabs->outPutTabWidget->getControlsBar());
  }
  if (!calibrationUpdateController) {
    calibrationUpdateController = new CalibrationUpdateController(projectTabs->outPutTabWidget->getControlsBar());
  }
  if (!inputPluginController) {
    inputPluginController = new InputPluginController(projectTabs->sourcesTabWidget);
  }
  if (!exposureController) {
    exposureController = new ExposureActionController(projectTabs->outPutTabWidget);
  }
  if (!outputPluginController) {
    outputPluginController = new OutputPluginController(projectTabs->configurationTabWidget->getConfigOutputs());
  }
  if (!cropInputController) {
    cropInputController.reset(new CropInputController(projectTabs->outPutTabWidget->getControlsBar()));
  }

  registerWidgetConnections();
}

void ProjectWorkWidget::initializeStitcher() {
  registerMetaTypes();
  registerStitcherSignals();
  registerSignalInjections();
  registerOutputSignals();
  registerControllers();
  startApplication();
}

void ProjectWorkWidget::registerWidgetConnections() {
  StitcherController* videoStitcher(GlobalController::getInstance().getController());
  LiveStitcherController* liveVideoStitcher = GlobalLiveController::getInstance().getController();

  connect(this, &ProjectWorkWidget::reqForceGPUInfoRefresh, &gpuInfoUpdater, &GPUInfoUpdater::refreshTick,
          Qt::UniqueConnection);
  connect(this, &ProjectWorkWidget::reqThreadQuit, &gpuInfoUpdater, &GPUInfoUpdater::quit, Qt::UniqueConnection);
  connect(this, &ProjectWorkWidget::reqThreadQuit, &stitcherControllerThread, &QThread::quit, Qt::UniqueConnection);
  connect(this, &ProjectWorkWidget::reqCancelKernelCompile, videoStitcher, &StitcherController::tryCancelBackendCompile,
          Qt::DirectConnection);
  connect(informationBar, &TopInformationBarWidget::notifyQuitProject, this, &ProjectWorkWidget::onCloseProject,
          Qt::UniqueConnection);
  connect(projectTabs, &MainTabWidget::currentChanged, this, &ProjectWorkWidget::updateClockForTab);
  connect(exposureController, &ExposureActionController::exposureActivationChanged, informationBar,
          &TopInformationBarWidget::onActivateExposure, Qt::UniqueConnection);
  connect(exposureController, &ExposureActionController::exposureActivationChanged, this,
          &ProjectWorkWidget::updateEnabilityAfterActivation, Qt::UniqueConnection);
  connect(exposureController, &ExposureActionController::exposureActivationChanged, this,
          &ProjectWorkWidget::setNeedToExtract, Qt::UniqueConnection);
  connect(outputsController, &OutputsController::orientationChanged,
          projectTabs->interactiveTabWidget->getInteractiveWidget(), &DeviceInteractiveWidget::setOrientation);
  connect(outputsController, &OutputsController::outputActivationChanged, this,
          &ProjectWorkWidget::updateEnabilityAfterActivation, Qt::UniqueConnection);
  connect(projectTabs->outPutTabWidget->getAudioProcessorWidget(), &AudioProcessorsWidget::notifyEditProcessor,
          liveVideoStitcher, &LiveStitcherController::updateAudioProcessor);
  connect(projectTabs->outPutTabWidget->getAudioProcessorWidget(), &AudioProcessorsWidget::notifyRemoveProcessor,
          liveVideoStitcher, &LiveStitcherController::removeAudioProcessor);
}

void ProjectWorkWidget::registerStitcherSignals() {
  StitcherController* videoStitcher(GlobalController::getInstance().getController());
  connect(videoStitcher, &StitcherController::projectInitialized, this, &ProjectWorkWidget::setProject);
  connect(videoStitcher, &StitcherController::stitcherClosed, outputPluginController,
          &OutputPluginController::onProjectClosed);
  connect(videoStitcher, &StitcherController::reqCleanStitcher, this, &ProjectWorkWidget::onCleanStitcher);
  connect(videoStitcher, &StitcherController::notifyStitcherReset, this, &ProjectWorkWidget::onStitcherReset);
  connect(videoStitcher, &StitcherController::reqEnableWindow, this, &ProjectWorkWidget::reqEnableWindow);
  connect(videoStitcher, &StitcherController::reqDisableWindow, this, &ProjectWorkWidget::reqDisableWindow);

  connect(videoStitcher, &StitcherController::reqCreateThumbnails, projectTabs->sourcesTabWidget->getSourcesWidget(),
          &SourceWidget::createThumbnails);
  connect(videoStitcher, &StitcherController::reqCreatePanoView, this, &ProjectWorkWidget::registerRenderer);

  connect(videoStitcher, &StitcherController::notifyErrorMessage, this, &ProjectWorkWidget::onStitcherErrorMessage);
  connect(videoStitcher, &StitcherController::notifyEndOfStream, this, &ProjectWorkWidget::onEndOfStreamReached);
  connect(videoStitcher, &StitcherController::reqResetDimensions, this, &ProjectWorkWidget::notifyPanoResized);
  connect(videoStitcher, &StitcherController::notifyBackendCompileProgress, this,
          &ProjectWorkWidget::notifyBackendCompileProgress);
  connect(videoStitcher, &StitcherController::notifyBackendCompileDone, this,
          &ProjectWorkWidget::notifyBackendCompileDone);
}

void ProjectWorkWidget::registerRenderer(std::vector<std::shared_ptr<VideoStitch::Core::PanoRenderer>>* renderers) {
  StitcherController* videoStitcher = GlobalController::getInstance().getController();
  DeviceVideoWidget* videoWidget = projectTabs->outPutTabWidget->getVideoWidget();
  DeviceInteractiveWidget* interactiveWidget = projectTabs->interactiveTabWidget->getInteractiveWidget();
  videoStitcher->lockedFunction([videoWidget, interactiveWidget, renderers]() {
    videoWidget->registerRenderer(renderers);
    interactiveWidget->registerRenderer(renderers);
  });
}

void ProjectWorkWidget::registerOutputSignals() {
  DeviceVideoWidget* videoWidget = projectTabs->outPutTabWidget->getVideoWidget();
  connect(videoWidget, &VideoWidget::rotatePanorama, GlobalController::getInstance().getController(),
          [](YPRSignalCaps* rotations) {
            GlobalController::getInstance().getController()->rotatePanorama(rotations, false);
          });
  connect(videoWidget, &DeviceVideoWidget::applyOrientation, GlobalLiveController::getInstance().getController(),
          &LiveStitcherController::selectOrientation);
}

void ProjectWorkWidget::registerSignalInjections() {
  StitcherController* videoStitcher = GlobalController::getInstance().getController();
  connect(this, &ProjectWorkWidget::reqOpenVAHFile, videoStitcher, &StitcherController::openProject);
  connect(this, &ProjectWorkWidget::reqCreateNewProject, GlobalLiveController::getInstance().getController(),
          &LiveStitcherController::createNewProject);
  connect(this, &ProjectWorkWidget::reqSave, videoStitcher, &StitcherController::saveProject);
  connect(this, &ProjectWorkWidget::notifyAudioPlaybackActivated, videoStitcher,
          &StitcherController::onActivateAudioPlayback);
}

void ProjectWorkWidget::registerControllers() {
  LiveStitcherController* liveVideoStitcher = GlobalLiveController::getInstance().getController();
  StitcherController* videoStitcher = GlobalController::getInstance().getController();

  // Calibration
  connect(calibrationActionController, &CalibrationActionController::reqApplyCalibrationImport, liveVideoStitcher,
          &LiveStitcherController::importCalibration, Qt::UniqueConnection);
  connect(calibrationActionController, &CalibrationActionController::reqApplyCalibrationTemplate, liveVideoStitcher,
          &LiveStitcherController::applyTemplate, Qt::UniqueConnection);
  connect(calibrationActionController, &CalibrationActionController::reqClearCalibration, liveVideoStitcher,
          &LiveStitcherController::clearCalibration);
  connect(videoStitcher, &StitcherController::notifyCalibrationStatus, calibrationActionController,
          &CalibrationActionController::onCalibrationImportError);
  connect(projectTabs->outPutTabWidget->getControlsBar()->buttonCalibrationToggleControlPoints, &QPushButton::clicked,
          liveVideoStitcher, &LiveStitcherController::toggleControlPoints);
  connect(calibrationUpdateController, &CalibrationUpdateController::reqCalibrationAdaptationProcess, liveVideoStitcher,
          &LiveStitcherController::onCalibrationAdaptationProcess);
  connect(calibrationUpdateController, &CalibrationUpdateController::reqResetPanoramaWithoutSave, liveVideoStitcher,
          &LiveStitcherController::onResetPanorama);

  // Crop
  connect(cropInputController.data(), &CropInputController::reqApplyCrops, videoStitcher,
          &StitcherController::applyCrops);

  // Exposure
  connect(exposureController, &ExposureActionController::reqClearExposure, liveVideoStitcher,
          &LiveStitcherController::clearExposure);
  connect(exposureController, &ExposureActionController::reqCompensateExposure, liveVideoStitcher,
          &LiveStitcherController::compensateExposure);
  connect(exposureController, &ExposureActionController::reqReplacePanorama, liveVideoStitcher,
          &LiveStitcherController::onResetPanorama, Qt::DirectConnection);

  // output plugin
  connect(outputPluginController, &OutputPluginController::reqAddOutput, liveVideoStitcher,
          &LiveStitcherController::insertOutput, Qt::UniqueConnection);
  connect(outputPluginController, &OutputPluginController::reqRemoveOutput, liveVideoStitcher,
          &LiveStitcherController::removeOutput, Qt::UniqueConnection);
  connect(outputPluginController, &OutputPluginController::reqUpdateOutput, liveVideoStitcher,
          &LiveStitcherController::updateOutput);

  // input plugin
  connect(inputPluginController, &InputPluginController::notifyConfigureInputs, liveVideoStitcher,
          &LiveStitcherController::configureInputsWith, Qt::UniqueConnection);
  connect(inputPluginController, &InputPluginController::notifyConfigureAudioInput, liveVideoStitcher,
          &LiveStitcherController::configureAudioInput, Qt::UniqueConnection);
  connect(inputPluginController, &InputPluginController::notifyTestActivated, liveVideoStitcher,
          &LiveStitcherController::testInputs, Qt::UniqueConnection);
  connect(liveVideoStitcher, &LiveStitcherController::notifyInputsConfigurationSuccess, inputPluginController,
          &InputPluginController::onConfiguringInputsSuccess, Qt::UniqueConnection);
  connect(liveVideoStitcher, &LiveStitcherController::notifyInputsConfigurationError, inputPluginController,
          &InputPluginController::onConfiguringInputsError, Qt::UniqueConnection);
  connect(liveVideoStitcher, &LiveStitcherController::notifyInputTested, inputPluginController,
          &InputPluginController::notifyTestResult);
  connect(liveVideoStitcher, &LiveStitcherController::notifyAudioInputNotFound, this,
          &ProjectWorkWidget::onAudioLoadError);

  // outputs
  connect(outputsController, &OutputsController::reqActivateOutput, liveVideoStitcher,
          &LiveStitcherController::toggleOutputActivation);
  connect(outputsController, &OutputsController::reqToggleOutput, liveVideoStitcher,
          &LiveStitcherController::toggleOutputActivation);
  connect(liveVideoStitcher, &LiveStitcherController::notifyOutputActivated, outputsController,
          &OutputsController::onOutputCreated, Qt::BlockingQueuedConnection);
  connect(liveVideoStitcher, &LiveStitcherController::notifyOutputRemoved, outputsController,
          &OutputsController::onWriterRemoved);
  connect(videoStitcher, &StitcherController::notifyOutputError, outputsController, &OutputsController::onOutputError);
  connect(liveVideoStitcher, &LiveStitcherController::notifyOutputTrying, outputsController,
          &OutputsController::onOutputTryingToActivate);
  connect(liveVideoStitcher, &LiveStitcherController::notifyOutputWriterCreated, outputsController,
          &OutputsController::onWriterCreated, Qt::BlockingQueuedConnection);
  connect(liveVideoStitcher, &LiveStitcherController::notifyOutputDisconnected, outputsController,
          &OutputsController::onOutputDisconnected);
  connect(liveVideoStitcher, &LiveStitcherController::notifyOutputConnected, outputsController,
          &OutputsController::onOutputConnected);
}

void ProjectWorkWidget::registerMetaTypes() {
  qRegisterMetaType<VideoStitch::Status>("VideoStitch::Status");
  qRegisterMetaType<VideoStitch::Ptv::Value*>("VideoStitch::Ptv::Value*");
  qRegisterMetaType<GUIStateCaps::State>("GUIStateCaps::State");
  qRegisterMetaType<QList<QUrl>>("QList<QUrl>");
  qRegisterMetaType<std::vector<std::string>>("std::vector<std::string>");
  qRegisterMetaType<std::string>("std::string");
  qRegisterMetaType<size_t>("size_t");
  qRegisterMetaType<size_t>("mtime_t");
  qRegisterMetaType<std::vector<std::tuple<VideoStitch::Input::VideoReader::Spec, bool, std::string>>>(
      "std::vector<std::tuple<VideoStitch::Input::VideoReader::Spec, bool, std::string> >");
  qRegisterMetaType<std::vector<std::vector<VideoStitch::Core::SourceSurface*>>*>(
      "std::vector<std::vector<VideoStitch::Core::SourceSurface*>>*");
  qRegisterMetaType<VideoStitch::Core::PanoDefinition*>("VideoStitch::Core::PanoDefinition*");
  qRegisterMetaType<LiveStitcherController::Callback>("LiveStitcherController::Callback");
  qRegisterMetaType<QMap<int, QString>>("QMap<int,QString>");
  qRegisterMetaType<SignalCompressionCaps*>("SignalCompressionCaps*");
  qRegisterMetaType<std::vector<VideoStitch::Ptv::Value*>>("std::vector<VideoStitch::Ptv::Value*>");
  qRegisterMetaType<Crop>("Crop");
  qRegisterMetaType<QVector<Crop>>("QVector<Crop>");
  qRegisterMetaType<InputLensClass::LensType>("InputLensClass::LensType");
  qRegisterMetaType<VideoStitch::Core::StereoRigDefinition::Orientation>(
      "VideoStitch::Core::StereoRigDefinition::Orientation");
  qRegisterMetaType<VideoStitch::Core::StereoRigDefinition::Geometry>(
      "VideoStitch::Core::StereoRigDefinition::Geometry");
  qRegisterMetaType<QVector<int>>("QVector<int>");
  qRegisterMetaType<std::shared_ptr<VideoStitch::Core::SourceRenderer>>(
      "std::shared_ptr<VideoStitch::Core::SourceRenderer>");
}

void ProjectWorkWidget::createStitcherController(QVector<int> devices) {
  Q_ASSERT(!devices.isEmpty());
  GlobalLiveController::getInstance().createController(devices[0]);
  stitcherControllerThread.setObjectName("Stitcher Controller Thread");
  GlobalController::getInstance().getController()->moveToThread(&stitcherControllerThread);
  GlobalLiveController::getInstance().getController()->moveToThread(&stitcherControllerThread);
  stitcherControllerThread.start();
  initializeMainTab();
  initializeStitcher();
  startOpenGL();
}

void ProjectWorkWidget::startOpenGL() {
  connect(projectTabs->outPutTabWidget->videoWidget, &DeviceVideoWidget::gotFrame, this,
          &ProjectWorkWidget::updateTopBar);
}

void ProjectWorkWidget::stopOpenGL() {
  disconnect(projectTabs->outPutTabWidget->videoWidget, &DeviceVideoWidget::gotFrame, this,
             &ProjectWorkWidget::updateTopBar);
  disconnect(projectTabs->outPutTabWidget, &OutPutTabWidget::refresh, nullptr, nullptr);
}

void ProjectWorkWidget::setProject(ProjectDefinition* project) {
  projectDefinition = qobject_cast<LiveProjectDefinition*>(project);
  projectTabs->sourcesTabWidget->setProject(projectDefinition);
  projectTabs->configurationTabWidget->setProject(projectDefinition);
  projectTabs->outPutTabWidget->setProject(projectDefinition);
  informationBar->setProject(projectDefinition);
  outputPluginController->setProject(projectDefinition);
  inputPluginController->setProject(projectDefinition);
  outputsController->setProject(projectDefinition);
  calibrationActionController->setProject(projectDefinition);
  calibrationUpdateController->setProject(projectDefinition);
  exposureController->setProject(projectDefinition);
  cropInputController->setProject(projectDefinition);

  if (projectDefinition->isInit()) {
    emit reqChangeState(GUIStateCaps::stitch);
    onPlay();
    emit reqForceGPUInfoRefresh();
    emit notifyProjectOpened();
  }
}

void ProjectWorkWidget::setNeedToExtract(bool newNeedToExtract) {
  if (newNeedToExtract) {
    ++needToExtract;
  } else {
    --needToExtract;
    Q_ASSERT_X(needToExtract >= 0, "ProjectWorkWidget::setNeedToExtract", "bad usage of needToExtract flag");
  }
  updateNextFrameAction();
}

void ProjectWorkWidget::onPlay() {
  GlobalController::getInstance().getController()->play();
  framerateComputer.start();
}

void ProjectWorkWidget::onPause() {
  auto controller = GlobalController::getInstance().getController();
  if (controller) {
    controller->pause();
  }
}

void ProjectWorkWidget::onCleanStitcher() {
  emit reqForceGPUInfoRefresh();
  emit reqChangeState(GUIStateCaps::idle);
}

void ProjectWorkWidget::updateTopBar(mtime_t date) {
  StitcherController* stitcherController = GlobalController::getInstance().getController();
  if (stitcherController) {
    if (stitcherController->isPlaying()) {
      framerateComputer.tick();
    }
    informationBar->updateCurrentTime(TimeConverter::dateToTimeDisplay(date), framerateComputer.getFramerate(),
                                      stitcherController->getFrameRate());
    if (projectDefinition->hasAudio()) {
      informationBar->updateVuMeterValues();
    }
  }
}

bool ProjectWorkWidget::outputIsActivated() const {
  return projectDefinition ? projectDefinition->areActiveOutputs() : false;
}

bool ProjectWorkWidget::algorithmIsActivated() const {
  return exposureController ? exposureController->exposureIsActivated() : false;
}

void ProjectWorkWidget::updateEnabilityAfterActivation() {
  bool outputIsActivated = projectDefinition->areActiveOutputs();
  bool algorithmIsActivated = exposureController->exposureIsActivated();
  projectTabs->outPutTabWidget->getControlsBar()->updateEditability(outputIsActivated, algorithmIsActivated);
  projectTabs->outPutTabWidget->getConfigPanoramaWidget()->updateEditability(outputIsActivated, algorithmIsActivated);
  projectTabs->configurationTabWidget->getConfigOutputs()->updateEditability(outputIsActivated, algorithmIsActivated);
  WidgetsManager::getInstance()->activateSourcesTab(!outputIsActivated && !algorithmIsActivated);
}

void ProjectWorkWidget::updateClockForTab(int tab) {
  switch (GuiEnums::Tab(tab)) {
    case GuiEnums::Tab::TabSources:
      nextFrameAction = StitcherController::NextFrameAction::StitchAndExtract;
      break;
    case GuiEnums::Tab::TabOutPut:
    case GuiEnums::Tab::TabInteractive:
    case GuiEnums::Tab::TabConfiguration:
      nextFrameAction = StitcherController::NextFrameAction::Stitch;
      break;
    default:
      nextFrameAction = StitcherController::NextFrameAction::None;
      break;
  }
  updateNextFrameAction();
  // FIXME: needs to repaint after switching tab.
  informationBar->repaint();
}

void ProjectWorkWidget::updateNextFrameAction() {
  StitcherController::NextFrameAction realNextFrameAction = nextFrameAction;
  if (needToExtract > 0) {
    realNextFrameAction = StitcherController::NextFrameAction(int(realNextFrameAction) |
                                                              int(StitcherController::NextFrameAction::Extract));
  }

  StitcherController* stitcherController = GlobalController::getInstance().getController();
  stitcherController->setNextFrameAction(realNextFrameAction);
}

void ProjectWorkWidget::onStitcherReset() {
  framerateComputer.restart();
  informationBar->setDefaultTime();
}

void ProjectWorkWidget::onStitcherErrorMessage(const VideoStitch::Status& status, bool needToExit) {
  GenericDialog* errorDialog = new GenericDialog(status, this);
  if (needToExit) {
    connect(errorDialog, &GenericDialog::notifyAcceptClicked, this, &ProjectWorkWidget::onCloseProject);
  }
  errorDialog->show();
}

void ProjectWorkWidget::onEndOfStreamReached() {
  GenericDialog* errorDialog =
      new GenericDialog(tr("End of stream reached"), tr("Could not load input frame, reader reported end of stream"),
                        GenericDialog::DialogMode::ACCEPT, this);
  connect(errorDialog, &GenericDialog::notifyAcceptClicked, this, &ProjectWorkWidget::onCloseProject);
  errorDialog->show();
}

void ProjectWorkWidget::onAudioLoadError(const QString& title, const QString& message) {
  GenericDialog* errorDialog(new GenericDialog(title, message, GenericDialog::DialogMode::ACCEPT_CANCEL, this));
  connect(errorDialog, &GenericDialog::notifyAcceptClicked, this, &ProjectWorkWidget::notifyRemoveAudio);
  connect(errorDialog, &GenericDialog::notifyCancelClicked, this, &ProjectWorkWidget::onCloseProject);
  connect(errorDialog, &GenericDialog::notifyCancelClicked, errorDialog, &GenericDialog::close);
  errorDialog->show();
}

void ProjectWorkWidget::onCloseProject() {
  // It's really important that the project and the stitcher controller are cleaned before returning from this function
  // In order to be able to open a new project

  // Begin the GUI cleaning
  StitcherController* videoStitcher = GlobalLiveController::getInstance().getController();
  videoStitcher->closingProject();
  stopOpenGL();
  onPause();
  emit reqChangeState(GUIStateCaps::idle);
  exposureController->toggleExposure(false);
  outputsController->clearProject();

  // Close the stitcher controller synchronously, in its own thread
  QMetaObject::invokeMethod(GlobalLiveController::getInstance().getController(), "closeProject",
                            Qt::BlockingQueuedConnection);
  QMetaObject::invokeMethod(GlobalController::getInstance().getController(), "resetProject",
                            Qt::BlockingQueuedConnection);

  // Finish the GUI cleaning
  projectDefinition = nullptr;
  projectTabs->sourcesTabWidget->getSourcesWidget()->clearThumbnails();
  projectTabs->sourcesTabWidget->clearProject();
  projectTabs->configurationTabWidget->clearProject();
  projectTabs->outPutTabWidget->clearProject();
  VSSettings::getSettings()->addRecentFile(ProjectFileHandler::getInstance()->getFilename());
  framerateComputer.stop();
  informationBar->clearProject();
  outputPluginController->clearProject();
  inputPluginController->clearProject();
  calibrationActionController->clearProject();
  calibrationUpdateController->clearProject();
  exposureController->clearProject();
  cropInputController->setProject(nullptr);
  // Finally, ask to delete the stitcher controller
  GlobalLiveController::getInstance().deleteController();

  emit notifyProjectClosed();
}
