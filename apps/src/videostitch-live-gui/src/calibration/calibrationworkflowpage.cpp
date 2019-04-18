// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationworkflowpage.hpp"
#include "ui_calibrationworkflowpage.h"

#include "calibrationsnapcounter.hpp"
#include "generic/workflowdialog.hpp"
#include "livesettings.hpp"
#include "videostitcher/globallivecontroller.hpp"
#include "widgetsmanager.hpp"

#include "libvideostitch-gui/mainwindow/vssettings.hpp"
#include "libvideostitch-gui/utils/timersecondsenum.hpp"
#include "libvideostitch-gui/videostitcher/presetsmanager.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/cameraDef.hpp"
#include "libvideostitch/rigDef.hpp"

CalibrationWorkflowPage::CalibrationWorkflowPage(QWidget* parent)
    : WorkflowPage(parent),
      ui(new Ui::CalibrationWorkflowPage),
      project(nullptr),
      oldFrameAction(int(StitcherController::NextFrameAction::None)) {
  ui->setupUi(this);
  ui->comboSeconds->addItems(SecondsEnum);
  ui->comboSeconds->setCurrentIndex(0);

  bool showCalibrationCounter = LiveSettings::getLiveSettings()->getShowCalibrationCounter();
  for (int index = 0; index < ui->counterLayout->count(); ++index) {
    if (ui->counterLayout->itemAt(index)->widget()) {
      ui->counterLayout->itemAt(index)->widget()->setVisible(showCalibrationCounter);
    }
  }
}

CalibrationWorkflowPage::~CalibrationWorkflowPage() {}

void CalibrationWorkflowPage::setProject(ProjectDefinition* p) {
  project = p;

  LiveStitcherController* liveVideoStitcher = GlobalLiveController::getInstance().getController();
  connect(this, &CalibrationWorkflowPage::reqCalibrate, liveVideoStitcher, &LiveStitcherController::calibrate,
          Qt::UniqueConnection);
  connect(this, &CalibrationWorkflowPage::reqResetPanorama, liveVideoStitcher, &LiveStitcherController::onResetPanorama,
          Qt::UniqueConnection);
}

void CalibrationWorkflowPage::initializePage() {
  StitcherController* stitcherController = GlobalController::getInstance().getController();
  assert(stitcherController->isPlaying());  // In Vahana VR, we should always play
  // To be able to calibrate, we need to extract the frames
  oldFrameAction = int(stitcherController->setNextFrameAction(StitcherController::NextFrameAction::StitchAndExtract));
}

void CalibrationWorkflowPage::deinitializePage() {
  StitcherController* stitcherController = GlobalController::getInstance().getController();
  stitcherController->setNextFrameAction(StitcherController::NextFrameAction(oldFrameAction));
}

void CalibrationWorkflowPage::save() {
  int seconds = ui->comboSeconds->currentText().toInt();
  if (seconds > 0) {
    VideoStitch::Logger::get(VideoStitch::Logger::Info)
        << "Launching calibration snap counter with: " << seconds << " seconds" << std::endl;
    CalibrationSnapCounter* calibrationSnapCounter =
        new CalibrationSnapCounter(WidgetsManager::getInstance()->getMainWindowRef());
    connect(calibrationSnapCounter, &CalibrationSnapCounter::notifyTimerEnded, this, [=]() {
      calibrationSnapCounter->deleteLater();
      calibrate();
    });
    calibrationSnapCounter->show();
    calibrationSnapCounter->startCounter(seconds);
  } else {
    calibrate();
  }
}

void CalibrationWorkflowPage::onPanorama(VideoStitch::Core::PanoramaDefinitionUpdater& pano) {
  VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Calibration done" << std::endl;
  emit reqResetPanorama(&pano);
  QMetaObject::invokeMethod(this, "onCalibrationSuccess", Qt::QueuedConnection);
}

void CalibrationWorkflowPage::onError(const VideoStitch::Status& error) {
  VideoStitch::Logger::get(VideoStitch::Logger::Warning)
      << "Failed to calibrate inputs: " << error.getErrorMessage() << std::endl;
  QMetaObject::invokeMethod(this, "onCalibrationFailure", Qt::QueuedConnection);
}

void CalibrationWorkflowPage::setAutoFov(bool tempUseAutoFov) { useAutoFov = tempUseAutoFov; }

void CalibrationWorkflowPage::calibrate() {
  std::unique_ptr<VideoStitch::Ptv::Value> calibrationConfig = buildCalibrationConfig();
  if (!calibrationConfig) {
    return;
  }
  VideoStitch::Potential<VideoStitch::Util::OnlineAlgorithm> status =
      VideoStitch::Util::OnlineAlgorithm::create("calibration", calibrationConfig.get());
  if (status.ok()) {
    VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Calibrating..." << std::endl;
    workflowDialog->showWaitingPage(tr("Calibrating..."));
    emit reqCalibrate(LiveStitcherController::Callback(status.release(), this, nullptr));
  } else {
    workflowDialog->showErrorMessage(tr("Could not launch the calibration algorithm"));
  }
}

std::unique_ptr<VideoStitch::Ptv::Value> CalibrationWorkflowPage::buildCalibrationConfig() const {
  // Load calibration configuration from preset
  std::unique_ptr<VideoStitch::Ptv::Value> calibrationConfig =
      PresetsManager::getInstance()->clonePresetContent("calibration", "default_vahana");
  if (!calibrationConfig) {
    workflowDialog->showErrorMessage(
        tr("Cannot initialize automatic calibration: there might be something wrong with your %0 installation")
            .arg(QCoreApplication::applicationName()));
    return std::unique_ptr<VideoStitch::Ptv::Value>();
  }

  calibrationConfig->get("improve_mode")->asBool() = false;  // start from scratch by default right now
  /*Replicate PTGui default behavior*/
  calibrationConfig->get("single_focal")->asBool() = true;
  calibrationConfig->get("dump_calibration_snapshots")->asBool() =
      VSSettings::getSettings()->getIsDumpingCalibrationPictures();

  if (project->getPanoConst()->hasCalibrationRigPresets()) {
    const VideoStitch::Core::RigDefinition& rigdef = project->getPanoConst()->getCalibrationRigPresets();
    calibrationConfig->push("rig", rigdef.serialize());
    VideoStitch::Ptv::Value* listCameras = VideoStitch::Ptv::Value::emptyObject();
    for (auto it : rigdef.getRigCameraDefinitionMap()) {
      listCameras->asList().push_back(it.second->serialize());
    }
    calibrationConfig->push("cameras", listCameras);
    calibrationConfig->get("deshuffle_mode")->asBool() = true;
    calibrationConfig->get("deshuffle_mode_preserve_readers_order")->asBool() = true;
  } else {
    calibrationConfig->get("initial_hfov")->asDouble() = project->getPanoConst()->getHFovFromInputSources();
    size_t nbCameras = project->getPanoConst()->numVideoInputs();
    const VideoStitch::Core::InputDefinition& firstVideoInput = project->getPanoConst()->getVideoInputs()[0];
    double hfov = project->getPanoConst()->getHFovFromInputSources();
    calibrationConfig->get("auto_iterate_fov")->asBool() =
        (useAutoFov || hfov <= 0.0);  // iterate over a predefined list of hfov values
    if (hfov <= 0.0) {
      hfov = PTV_DEFAULT_HFOV;
    }

    std::unique_ptr<VideoStitch::Core::RigDefinition> rigdef(VideoStitch::Core::RigDefinition::createBasicUnknownRig(
        "default", project->getPanoConst()->getLensFormatFromInputSources(), nbCameras, firstVideoInput.getWidth(),
        firstVideoInput.getHeight(), firstVideoInput.getCroppedWidth(), firstVideoInput.getCroppedHeight(), hfov,
        project->getPanoConst().get()));
    calibrationConfig->push("rig", rigdef->serialize());

    VideoStitch::Ptv::Value* listCameras = VideoStitch::Ptv::Value::emptyObject();
    for (auto it : rigdef->getRigCameraDefinitionMap()) {
      listCameras->asList().push_back(it.second->serialize());
    }
    calibrationConfig->push("cameras", listCameras);
    calibrationConfig->get("deshuffle_mode")->asBool() = false;
  }

  // TODO for incremental calibration: have an increment instead of the 0 default value
  VideoStitch::Ptv::Value* listFrames = VideoStitch::Ptv::Value::emptyObject();
  VideoStitch::Ptv::Value* calibrationFrameID = VideoStitch::Ptv::Value::emptyObject();
  calibrationFrameID->asInt() = 0;
  listFrames->asList().push_back(calibrationFrameID);
  calibrationConfig->push("list_frames", listFrames);
  return calibrationConfig;
}

void CalibrationWorkflowPage::onCalibrationSuccess() {
  workflowDialog->completeCurrentPage(tr("Calibration done"));
  workflowDialog->closeWaitingPage();
  emit calibrationChanged();
}

void CalibrationWorkflowPage::onCalibrationFailure() {
  workflowDialog->showErrorMessage(
      tr("The calibration process has failed. Please, rotate your rig and repeat the process."));
  workflowDialog->closeWaitingPage();
}
