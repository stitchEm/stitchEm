// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exposureactioncontroller.hpp"
#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/inputDef.hpp"
#include "guiconstants.hpp"
#include "outputtabwidget.hpp"
#include "widgetsmanager.hpp"
#include "generic/genericdialog.hpp"
#include "configurations/configexposurewidget.hpp"
#include "liveexposure.hpp"

ExposureActionController::ExposureActionController(OutPutTabWidget* widget)
    : controlsPanel(widget->getControlsBar()), expoContTimer(this), projectDefinition(nullptr), outputTabRef(widget) {
  connect(this, &ExposureActionController::reqCancelExposure, this, &ExposureActionController::onExposureFailed);
  connect(&expoContTimer, &QTimer::timeout, this, &ExposureActionController::onExposureApplied);
  connect(controlsPanel->buttonExposureApply, &QPushButton::toggled, this,
          &ExposureActionController::onActivationFromControlBar);
  connect(controlsPanel->buttonExposureClear, &QPushButton::clicked, this, &ExposureActionController::reqClearExposure);
  connect(controlsPanel->buttonExposureSettings, &QPushButton::clicked, this,
          &ExposureActionController::onExposureSettings);
}

ExposureActionController::~ExposureActionController() { expoContTimer.stop(); }

// This function is called when the exposure compensation algorithm has finished and returned a new panorama
void ExposureActionController::onPanorama(VideoStitch::Core::PanoramaDefinitionUpdater& pano) {
  emit reqReplacePanorama(&pano);
}

void ExposureActionController::onError(const VideoStitch::Status&) { emit reqCancelExposure(); }

bool ExposureActionController::exposureIsActivated() const { return expoContTimer.isActive(); }

void ExposureActionController::startExposure() {
  expoContTimer.start(EXPO_TIMEOUT);
  emit exposureActivationChanged(true);
}

void ExposureActionController::stopExposure() {
  expoContTimer.stop();
  emit exposureActivationChanged(false);
}

void ExposureActionController::showExposureErrorMessage() {
  QString algorithm = projectDefinition->getDelegate()->getExposureConfiguration().getAlgorithm();
  GenericDialog* errorDialog = new GenericDialog(
      tr("Exposure compensation Error"),
      tr("Could not apply exposure compensation for algorithm: %0.\nCancelling operation").arg(algorithm),
      GenericDialog::DialogMode::ACCEPT, WidgetsManager::getInstance()->getMainWindowRef());
  errorDialog->show();
  errorDialog->raise();
}

void ExposureActionController::onExposureApplied() {
  if (!projectDefinition) {
    return;
  }
  LiveExposure& exposureConfig = projectDefinition->getDelegate()->getExposureConfiguration();
  std::unique_ptr<VideoStitch::Ptv::Value> exposureAlgorithm(VideoStitch::Ptv::Value::emptyObject());
  exposureAlgorithm->get("stabilize_wb")->asBool() = true;
  exposureAlgorithm->get("anchor")->asInt() = exposureConfig.getAnchor();
  VideoStitch::Potential<VideoStitch::Util::OnlineAlgorithm> fStatusExpo =
      VideoStitch::Util::OnlineAlgorithm::create(exposureConfig.getAlgorithm().toStdString(), exposureAlgorithm.get());
  if (!fStatusExpo.ok()) {
    toggleExposure(false);
    showExposureErrorMessage();
    return;
  }
  emit reqCompensateExposure(LiveStitcherController::Callback(fStatusExpo.release(), this, nullptr));
}

void ExposureActionController::onExposureFailed() {
  // TODO: think what will be the bahavior if the algorithm fails.
  //      In this case we keep retrying even if it fails until the user stops it
  //      Ahother solution will we to define a max number retries
}

void ExposureActionController::setProject(ProjectDefinition* project) {
  projectDefinition = qobject_cast<LiveProjectDefinition*>(project);
  checkAutoExposure();
}

void ExposureActionController::clearProject() { projectDefinition = nullptr; }

void ExposureActionController::onExposureSettings() {
  ConfigExposureWidget* settingsDialog =
      new ConfigExposureWidget(WidgetsManager::getInstance()->currentTab(), projectDefinition);
  settingsDialog->show();
}

void ExposureActionController::onActivationFromControlBar(const bool active) {
  if (active) {
    startExposure();
  } else {
    stopExposure();
  }
}

void ExposureActionController::toggleExposure(const bool active) {
  controlsPanel->buttonExposureApply->setChecked(active);
}

void ExposureActionController::checkAutoExposure() {
  if (projectDefinition && projectDefinition->isInit()) {
    if (projectDefinition->getDelegate()->getExposureConfiguration().getIsAutoStart()) {
      WidgetsManager::getInstance()->changeTab(GuiEnums::Tab::TabOutPut);
      toggleExposure(true);
    }
  }
}
