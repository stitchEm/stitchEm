// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rigworkflowpage.hpp"
#include "ui_rigworkflowpage.h"

#include "generic/workflowdialog.hpp"
#include "videostitcher/globallivecontroller.hpp"

#include "libvideostitch-gui/base/ptvMerger.hpp"
#include "libvideostitch-gui/mainwindow/outputfilehandler.hpp"
#include "libvideostitch-gui/mainwindow/vssettings.hpp"
#include "libvideostitch-gui/videostitcher/presetsmanager.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/cameraDef.hpp"
#include "libvideostitch/rigDef.hpp"

#include <QFileDialog>

RigWorkflowPage::RigWorkflowPage(QWidget* parent) : WorkflowPage(parent), ui(new Ui::RigWorkflowPage) {
  ui->setupUi(this);
  ui->radioAutoCalibration->setChecked(true);
  ui->rigWidget->setVisible(true);
  ui->templateSelection->setVisible(false);
  connect(ui->buttonBrowseTemplate, &QPushButton::clicked, this, &RigWorkflowPage::onButtonImportClicked);
  connect(ui->radioAutoCalibration, &QRadioButton::toggled, this, [&](bool check) {
    ui->rigWidget->setVisible(check);
    ui->templateSelection->setVisible(!check);
    workflowDialog->blockNextStep(!check && ui->lineTemplateName->text().isEmpty());
  });
}

RigWorkflowPage::~RigWorkflowPage() {}

void RigWorkflowPage::setProject(ProjectDefinition* p) {
  LiveStitcherController* liveVideoStitcher = GlobalLiveController::getInstance().getController();
  connect(this, &RigWorkflowPage::reqCalibrate, liveVideoStitcher, &LiveStitcherController::calibrate,
          Qt::UniqueConnection);
  connect(this, &RigWorkflowPage::reqReset, liveVideoStitcher, &LiveStitcherController::reset, Qt::UniqueConnection);
  connect(this, &RigWorkflowPage::reqResetPanorama, liveVideoStitcher, &LiveStitcherController::onResetPanorama,
          Qt::UniqueConnection);
  connect(this, &RigWorkflowPage::reqSaveProject, GlobalController::getInstance().getController(),
          &StitcherController::saveProject, Qt::UniqueConnection);
  ui->rigWidget->setProject(p);
}

void RigWorkflowPage::save() {
  if (ui->radioImportTemplate->isChecked()) {
    // apply template
    const QString path = ui->lineTemplateName->text();
    if (!path.isEmpty()) {
      if (File::getTypeFromFile(path) == File::CALIBRATION) {
        workflowDialog->showWaitingPage(tr("Applying template..."));
        emit reqApplyCalibrationImport(path);
      } else if (File::getTypeFromFile(path) == File::PTV || File::getTypeFromFile(path) == File::VAH) {
        workflowDialog->showWaitingPage(tr("Applying template..."));
        emit reqApplyCalibrationTemplate(path);
      } else {
        workflowDialog->showErrorMessage(
            tr("Template file is not compatible with %0").arg(QCoreApplication::applicationName()));
      }
      // TODO: report template result in workflow
      workflowDialog->close();
    }
  } else {
    // apply custom rig
    if (ui->rigWidget->customRigIsSelected()) {
      ui->rigWidget->getProject()->getPano()->setCalibrationRigPresets(new VideoStitch::Core::RigDefinition());
      VideoStitch::Core::InputDefinition::Format format = InputLensClass::getInputDefinitionFormatFromLensType(
          ui->rigWidget->getCurrentLensType(),
          ui->rigWidget->getProject()->getPanoConst()->getLensModelCategoryFromInputSources());
      double hfov = ui->rigWidget->getHfov();
      emit useAutoFov(hfov <= 0.0);
      if (hfov <= 0.0) {
        hfov = PTV_DEFAULT_HFOV;
      }
      VideoStitch::Logger::get(VideoStitch::Logger::Info)
          << "Applying rig parameters, lens type = "
          << InputLensEnum::getDescriptorFromEnum(ui->rigWidget->getCurrentLensType()).toStdString()
          << ", hfov = " << hfov << std::endl;

      for (VideoStitch::Core::InputDefinition& inputDef : ui->rigWidget->getProject()->getPano()->getVideoInputs()) {
        inputDef.setFormat(format);
        inputDef.resetGeometries(hfov);
      }

      emit reqReset();
      emit reqSaveProject(ProjectFileHandler::getInstance()->getFilename());
      workflowDialog->completeCurrentPage(tr("Rig parameters applied successfully"));
    } else {
      // apply calibration with rig preset
      std::shared_ptr<VideoStitch::Ptv::Value> calibrationConfig(
          PresetsManager::getInstance()->clonePresetContent("calibration", "default_vahana").release());
      if (calibrationConfig == nullptr) {
        workflowDialog->showErrorMessage(
            tr("Cannot initialize automatic calibration: there might be something wrong with your %0 installation")
                .arg(QCoreApplication::applicationName()) +
            "\n" + tr("Please reinstall %0 if the problem persists.").arg(QCoreApplication::applicationName()));
        return;
      }

      calibrationConfig->get("improve_mode")->asBool() = false;
      calibrationConfig->get("apply_presets_only")->asBool() = true;
      /*Replicate PTGui default behavior*/
      calibrationConfig->get("single_focal")->asBool() = true;
      calibrationConfig->get("dump_calibration_snapshots")->asBool() =
          VSSettings::getSettings()->getIsDumpingCalibrationPictures();

      std::unique_ptr<VideoStitch::Ptv::Value> rigPresetValue = ui->rigWidget->cloneSelectedRigPreset();
      VideoStitch::Helper::PtvMerger::mergeValue(calibrationConfig.get(), rigPresetValue.get());

      VideoStitch::Potential<VideoStitch::Util::OnlineAlgorithm> status =
          VideoStitch::Util::OnlineAlgorithm::create("calibration", calibrationConfig.get());
      if (status.ok()) {
        VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Applying rig preset..." << std::endl;
        workflowDialog->showWaitingPage(tr("Applying rig preset..."));
        emit reqCalibrate(LiveStitcherController::Callback(status.release(), this, nullptr));
      } else {
        workflowDialog->showErrorMessage(tr("Failed to apply rig preset"));
      }
    }
  }
}

void RigWorkflowPage::onPanorama(VideoStitch::Core::PanoramaDefinitionUpdater& pano) {
  VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Rig preset applied" << std::endl;

  // Add the rig preset in the pano
  std::unique_ptr<VideoStitch::Ptv::Value> rigPresetValue = ui->rigWidget->cloneSelectedRigPreset();
  std::map<std::string, std::shared_ptr<VideoStitch::Core::CameraDefinition> > cameraMap;
  const VideoStitch::Ptv::Value* cameraListValue = rigPresetValue->has("cameras");
  if (cameraListValue && cameraListValue->getType() == VideoStitch::Ptv::Value::LIST) {
    for (auto value : cameraListValue->asList()) {
      std::shared_ptr<VideoStitch::Core::CameraDefinition> camera(VideoStitch::Core::CameraDefinition::create(*value));
      if (camera.get()) {
        cameraMap[camera->getName()] = camera;
      }
    }
  }

  if (!cameraMap.empty()) {
    // Load rig presets
    const VideoStitch::Ptv::Value* rigValue = rigPresetValue->has("rig");
    if (rigValue && rigValue->getType() == VideoStitch::Ptv::Value::OBJECT) {
      pano.setCalibrationRigPresets(VideoStitch::Core::RigDefinition::create(cameraMap, *rigValue));
    }
  }

  connect(GlobalController::getInstance().getController(), &StitcherController::projectInitialized, this,
          &RigWorkflowPage::onRigPresetAppliedSuccessfully);
  emit reqResetPanorama(&pano);
}

void RigWorkflowPage::onError(const VideoStitch::Status& error) {
  VideoStitch::Logger::get(VideoStitch::Logger::Warning)
      << "Failed to apply rig preset: " << error.getErrorMessage() << std::endl;
  QMetaObject::invokeMethod(this, "onRigPresetFailure", Qt::QueuedConnection);
}

void RigWorkflowPage::onRigPresetAppliedSuccessfully() {
  disconnect(GlobalController::getInstance().getController(), &StitcherController::projectInitialized, this,
             &RigWorkflowPage::onRigPresetAppliedSuccessfully);
  workflowDialog->completeCurrentPage(tr("Rig preset applied"));
  workflowDialog->closeWaitingPage();
  emit rigChanged();
}

void RigWorkflowPage::onRigPresetFailure() {
  workflowDialog->showErrorMessage(tr("Failed to apply rig preset"));
  workflowDialog->closeWaitingPage();
}

void RigWorkflowPage::onButtonImportClicked() {
  const QString dir =
      VSSettings::getSettings()
          ->getValue("gui/calibration-last-path", ProjectFileHandler::getInstance()->getWorkingDirectory())
          .toString();
  const QString& path = QFileDialog::getOpenFileName(this, tr("Select a calibration template"), dir,
                                                     tr("Template (*.ptv *.ptvb *.pto *.pts *.vah)"));
  if (!path.isEmpty()) {
    VSSettings::getSettings()->setValue("gui/calibration-last-path", QFileInfo(path).absoluteDir().absolutePath());
    ui->lineTemplateName->setText(path);
    workflowDialog->blockNextStep(false);
  }
}
