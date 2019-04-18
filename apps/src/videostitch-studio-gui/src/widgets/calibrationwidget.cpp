// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationwidget.hpp"
#include "ui_calibrationwidget.h"

#include "commands/calibrationappliedcommand.hpp"
#include "commands/panochangedcommand.hpp"
#include "widgets/photometriccalibrationwidget.hpp"
#include "videostitcher/globalpostprodcontroller.hpp"

#include "libvideostitch-gui/base/ptvMerger.hpp"
#include "libvideostitch-gui/widgets/crop/cropwindow.hpp"
#include "libvideostitch-gui/mainwindow/LibLogHelpers.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/ui_header/progressreporterwrapper.hpp"
#include "libvideostitch-gui/mainwindow/vssettings.hpp"
#include "libvideostitch-gui/utils/imagesorproceduralsonlyfilterer.hpp"
#include "libvideostitch-gui/utils/onevisualinputfilterer.hpp"
#include "libvideostitch-gui/utils/visibilityeventfilterer.hpp"
#include "libvideostitch-gui/videostitcher/presetsmanager.hpp"

#include "libvideostitch-base/vslog.hpp"

#include "libvideostitch/rigDef.hpp"
#include "libvideostitch/rigCameraDef.hpp"
#include "libvideostitch/cameraDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/logging.hpp"

#include <QMenu>
#include <fstream>
#include <set>

CalibrationWidget::CalibrationWidget(QWidget* const parent)
    : ComputationWidget(parent), ui(new Ui::CalibrationWidget), currentFrame(0), algorithmType(AlgorithmType::None) {
  ui->setupUi(this);

  // Add the event filters first
  OneVisualInputFilterer::getInstance()->watch(ui->automaticCalibStackedWidget,
                                               FeatureFilterer::PropertyToWatch::enabled);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->framesSelected, FeatureFilterer::PropertyToWatch::visible);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->improveOnCurrentFrameButton,
                                                        FeatureFilterer::PropertyToWatch::visible);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->boxFrames, FeatureFilterer::PropertyToWatch::visible);

  // These initializations are required to ensure to show a proper widget at startup
  ui->stackedWidget->setCurrentIndex(0);
  ui->automaticCalibStackedWidget->setCurrentWidget(ui->pageParameters);
  ui->stackedImport->setCurrentWidget(ui->pageSelection);
  ui->radioFrameAutomatic->setChecked(true);
  ui->radioFrameManual->setChecked(false);
  ui->removeFrameButton->setEnabled(false);
  ui->clearFramesButton->setEnabled(false);
  ui->manualInputFramesWidget->setVisible(false);
  ui->deshuffleButton->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  ui->deshuffleOnlyOnListButton->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  ui->deshuffleOnlyOnCurrentFrameButton->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  ui->syntheticKeypointsCheckBox->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  ui->syntheticKeypointsCheckBox_InImprove->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  ui->recentCalibrationsButton->setMenu(new QMenu(ui->recentCalibrationsButton));
  ui->checkDisplayFrames->setChecked(false);
  ui->radioFrameAutomatic->setAttribute(Qt::WA_LayoutUsesWidgetRect);
  ui->radioFrameManual->setAttribute(Qt::WA_LayoutUsesWidgetRect);
  ui->listUsedFrames->setVisible(false);

  connect(ui->applyCalibrationButton, &QPushButton::clicked, this, &CalibrationWidget::reqImportCalibration);
  connect(ui->cropButton, &QPushButton::clicked, this, &CalibrationWidget::adjustCrop);
  connect(ui->cropButton_inImprove, &QPushButton::clicked, this, &CalibrationWidget::adjustCrop);
  connect(ui->calibrateButton, &QPushButton::clicked, this, &CalibrationWidget::startCalibrationOnList);
  connect(ui->rigWidget, &RigWidget::currentRigChanged, this, &CalibrationWidget::updateRigRelatedWidgets);
  connect(ui->deshuffleButton, &QPushButton::clicked, this, &CalibrationWidget::deshuffle);
  connect(ui->deshuffleOnlyOnListButton, &QPushButton::clicked, this, &CalibrationWidget::deshuffleOnlyOnList);
  connect(ui->deshuffleOnlyOnCurrentFrameButton, &QPushButton::clicked, this,
          &CalibrationWidget::deshuffleOnlyOnCurrentFrame);
  connect(ui->improveOnCurrentFrameButton, &QPushButton::clicked, this,
          &CalibrationWidget::improveCalibrationOnCurrentFrame);
  connect(ui->resetButton, &QPushButton::clicked, this, &CalibrationWidget::resetCalibration);
  connect(ui->buttonResetTemplate, &QPushButton::clicked, this, &CalibrationWidget::resetCalibration);
  connect(ui->addFrameButton, &QPushButton::clicked, this, &CalibrationWidget::addCurrentFrame);
  connect(ui->removeFrameButton, &QPushButton::clicked, this, &CalibrationWidget::removeFrame);
  connect(ui->clearFramesButton, &QPushButton::clicked, this, &CalibrationWidget::clearFrames);
  connect(ui->listFrames, &QListWidget::itemDoubleClicked, this, &CalibrationWidget::seekFrame);
  connect(ui->listFrames, &QListWidget::itemSelectionChanged, this, &CalibrationWidget::frameItemSelected);
  connect(ui->listUsedFrames, &QListWidget::itemDoubleClicked, this, &CalibrationWidget::seekFrame);
  connect(ui->syntheticKeypointsCheckBox, &QCheckBox::stateChanged, this,
          &CalibrationWidget::generateSyntheticKeypointsStateChanged);
  connect(ui->syntheticKeypointsCheckBox_InImprove, &QCheckBox::stateChanged, this,
          &CalibrationWidget::generateSyntheticKeypointsStateChanged);
  connect(ui->radioAutomatic, &QRadioButton::toggled, this, &CalibrationWidget::calibrationModeChanged);
  connect(ui->radioFrameAutomatic, &QRadioButton::toggled, this, &CalibrationWidget::calibrationFrameSelectionModes);
  connect(ui->checkDisplayFrames, &QCheckBox::toggled, this, [&](bool show) { ui->listUsedFrames->setVisible(show); });
}

CalibrationWidget::~CalibrationWidget() {}

void CalibrationWidget::fillRecentCalibrationMenuWith(QList<QAction*> recentCalibrationActions) {
  QMenu* recentMenu = ui->recentCalibrationsButton->menu();
  bool hasItems = false;
  foreach (QAction* action, recentCalibrationActions) {
    recentMenu->insertAction(0, action);
    hasItems = hasItems || !action->text().isEmpty();
  }
  ui->recentCalibrationsButton->setEnabled(hasItems);
}

void CalibrationWidget::applyCalibration(VideoStitch::Core::PanoDefinition* panoDef) {
  emit reqApplyCalibration(panoDef);
  updateWhenCalibrationChanged(panoDef);
}

void CalibrationWidget::startCalibrationOnList() {
  //: Warning popup title
  auto result =
      project->hasImagesOrProceduralsOnly() || project->getPanoConst()->hasBeenSynchronized()
          ? QMessageBox::Yes
          : MsgBoxHandler::getInstance()->genericSync(tr("Your videos might not be synchronized, continue anyway?"),
                                                      //: Warning popup text
                                                      tr("Warning"), WARNING_ICON, QMessageBox::Yes | QMessageBox::No);
  if (result == QMessageBox::Yes) {
    /* start new calibration */
    VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Starting calibration algorithm" << std::endl;
    startComputation(int(CalibrationOption::NoOption));
  }
}

void CalibrationWidget::deshuffleOnlyOnList() {
  /* start deshuffling only on frame list */
  VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Starting deshuffling algorithm" << std::endl;
  startComputation(int(CalibrationOption::DeshuffleOnly));
}

void CalibrationWidget::deshuffleOnlyOnCurrentFrame() {
  /* start deshuffling only on current frame */
  VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Starting deshuffling on current frame" << std::endl;
  startComputation(int(CalibrationOption::DeshuffleOnly) | int(CalibrationOption::ImproveGeometricCalibrationMode));
}

void CalibrationWidget::improveCalibrationOnCurrentFrame() {
  /* improve existing calibration */
  VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Starting improving calibration" << std::endl;
  startComputation(int(CalibrationOption::ImproveGeometricCalibrationMode));
}

void CalibrationWidget::attachListOfFramesToConfig(std::shared_ptr<VideoStitch::Ptv::Value> config,
                                                   bool improveGeometricCalibrationMode) const {
  QSet<int> frames;

  if (improveGeometricCalibrationMode || project->hasImagesOrProceduralsOnly()) {
    frames.insert(currentFrame);
  } else {
    if (ui->radioFrameAutomatic->isChecked()) {
      static int nbSteps = 10;
      int step = (project->getLastFrame() - project->getFirstFrame()) / nbSteps;
      for (int stepIndex = 0; stepIndex < nbSteps; ++stepIndex) {
        frames.insert(project->getFirstFrame() + stepIndex * step);
      }
      // Workaround for VSA-5907 (see also VSA-5887)
      if (project->getLastFrame() !=
          GlobalPostProdController::getInstance().getController()->getLastStitchableFrame()) {
        frames.insert(project->getLastFrame());
      }
    } else {
      for (int row = 0; row < ui->listFrames->count(); row++) {
        QListWidgetItem* item = ui->listFrames->item(row);
        if (item->data(Qt::UserRole).canConvert<int>()) {
          frames.insert(item->data(Qt::UserRole).toInt());
        }
      }
    }
  }

  std::vector<VideoStitch::Ptv::Value*>& dest = config->get("list_frames")->asList();
  for (int frame : frames) {
    VideoStitch::Ptv::Value* val = VideoStitch::Ptv::Value::emptyObject();
    val->asInt() = frame;
    dest.push_back(val);
  }
}

void CalibrationWidget::updateWhenCalibrationChanged(const VideoStitch::Core::PanoDefinition* pano) {
  const bool hasGeometries = pano->hasBeenCalibrated();
  const bool hasCP = pano->hasCalibrationControlPoints();

  showCalibrationModes(!hasGeometries);
  ui->stackedWidget->setCurrentWidget(hasGeometries && !hasCP ? ui->pageImport : ui->pageAutomatic);
  ui->automaticCalibStackedWidget->setCurrentWidget(hasGeometries ? ui->pageResults : ui->pageParameters);
  ui->stackedImport->setCurrentWidget(hasGeometries ? ui->pageImportResults : ui->pageSelection);
  if (hasCP) {
    ui->rigWidget->updateWhenCalibrationChanged(pano);
    const QString rigName = pano->hasCalibrationRigPresets()
                                ? QString::fromStdString(pano->getCalibrationRigPresetsName())
                                : RigWidget::customRigName();
    //: Message displayed after a calibration. %0 is the rig name
    ui->labelUsedRig->setText(tr("Rig used for calibration: %0").arg(rigName));
    std::set<int> usedFrames;  // To sort the used frames
    for (const auto& controlPoint : pano->getCalibrationControlPointList()) {
      usedFrames.insert(controlPoint.frameNumber);
    }

    ui->listUsedFrames->clear();
    for (int frame : usedFrames) {
      QListWidgetItem* frameItem = new QListWidgetItem(ui->listUsedFrames);
      //: Frame description in the result of the calibration. %0 is the frame number
      frameItem->setText(tr("Frame %0").arg(frame));
      frameItem->setData(Qt::UserRole, QVariant(frame));
      ui->listUsedFrames->addItem(frameItem);
    }
    ui->framesSelected->setVisible(!usedFrames.empty());
  }
}

std::shared_ptr<VideoStitch::Ptv::Value> CalibrationWidget::buildCalibrationConfig(int calibrationOptions) const {
  std::shared_ptr<VideoStitch::Ptv::Value> calibrationConfig(
      PresetsManager::getInstance()->clonePresetContent("calibration", "default_studio").release());
  if (calibrationConfig == nullptr) {
    MsgBoxHandler::getInstance()->generic(
        tr("Cannot initialize automatic calibration: there might be something wrong with your %0 installation")
                .arg(QCoreApplication::applicationName()) +
            "\n" + tr("Please reinstall %0 if the problem persists.").arg(QCoreApplication::applicationName()),
        tr("Missing default presets"), CRITICAL_ERROR_ICON);
    return std::shared_ptr<VideoStitch::Ptv::Value>();
  }

  calibrationConfig->get("improve_mode")->asBool() =
      calibrationOptions & int(CalibrationOption::ImproveGeometricCalibrationMode);
  calibrationConfig->get("apply_presets_only")->asBool() = calibrationOptions & int(CalibrationOption::ApplyPresetOnly);
  /*Replicate PTGui default behavior*/
  calibrationConfig->get("single_focal")->asBool() = true;
  calibrationConfig->get("use_synthetic_keypoints")->asBool() =
      (ui->syntheticKeypointsCheckBox->checkState() == Qt::CheckState::Checked);
  calibrationConfig->get("dump_calibration_snapshots")->asBool() =
      VSSettings::getSettings()->getIsDumpingCalibrationPictures();

  // Deshuffle input videos if option was passed
  calibrationConfig->get("deshuffle_mode")->asBool() = !ui->rigWidget->customRigIsSelected();
  calibrationConfig->get("deshuffle_mode_only")->asBool() = calibrationOptions & int(CalibrationOption::DeshuffleOnly);
  calibrationConfig->get("deshuffle_mode_preserve_readers_order")->asBool() = true;

  if ((calibrationOptions & int(CalibrationOption::ImproveGeometricCalibrationMode)) == 0) {  // Normal calibration
    if (!ui->rigWidget->customRigIsSelected()) {
      // Deshuffle input videos if !ApplyPresetsOnly and !ImproveGeometricCalibrationMode
      calibrationConfig->get("deshuffle_mode")->asBool() =
          !(calibrationOptions & int(CalibrationOption::ApplyPresetOnly));

      std::unique_ptr<VideoStitch::Ptv::Value> rigPresetValue = ui->rigWidget->cloneSelectedRigPreset();
      VideoStitch::Helper::PtvMerger::mergeValue(calibrationConfig.get(), rigPresetValue.get());

      // Add the rig preset in the pano
      std::map<std::string, std::shared_ptr<VideoStitch::Core::CameraDefinition> > cameraMap;
      const VideoStitch::Ptv::Value* cameraListValue = rigPresetValue->has("cameras");
      if (cameraListValue && cameraListValue->getType() == VideoStitch::Ptv::Value::LIST) {
        for (auto value : cameraListValue->asList()) {
          std::shared_ptr<VideoStitch::Core::CameraDefinition> camera(
              VideoStitch::Core::CameraDefinition::create(*value));
          if (camera.get()) {
            cameraMap[camera->getName()] = camera;
          }
        }
      }

      if (!cameraMap.empty()) {
        // Load rig presets
        const VideoStitch::Ptv::Value* rigValue = rigPresetValue->has("rig");
        if (rigValue && rigValue->getType() == VideoStitch::Ptv::Value::OBJECT) {
          panoDef->setCalibrationRigPresets(VideoStitch::Core::RigDefinition::create(cameraMap, *rigValue));
        }
      }
    } else {
      bool success = addCustomRigToCalibrationConfig(
          calibrationConfig, ui->rigWidget->getHfov(),
          InputLensClass::getInputDefinitionFormatFromLensType(ui->rigWidget->getCurrentLensType(),
                                                               panoDef->getLensModelCategoryFromInputSources()));
      if (!success) {
        return std::shared_ptr<VideoStitch::Ptv::Value>();
      }
      panoDef->setCalibrationRigPresets(new VideoStitch::Core::RigDefinition());
    }
  } else {  // Improve calibration mode
    if (panoDef->hasCalibrationRigPresets()) {
      const VideoStitch::Core::RigDefinition& rigdef = panoDef->getCalibrationRigPresets();
      calibrationConfig->push("rig", rigdef.serialize());

      VideoStitch::Ptv::Value* listCameras = VideoStitch::Ptv::Value::emptyObject();
      for (auto it : rigdef.getRigCameraDefinitionMap()) {
        listCameras->asList().push_back(it.second->serialize());
      }
      calibrationConfig->push("cameras", listCameras);
    } else {
      bool success = addCustomRigToCalibrationConfig(calibrationConfig, panoDef->getHFovFromInputSources(),
                                                     panoDef->getLensFormatFromInputSources());
      if (!success) {
        return std::shared_ptr<VideoStitch::Ptv::Value>();
      }
    }

    // add calibration control points
    calibrationConfig->push("calibration_control_points", panoDef->getControlPointListDef().serialize());
  }

  attachListOfFramesToConfig(calibrationConfig,
                             calibrationOptions & int(CalibrationOption::ImproveGeometricCalibrationMode));

  // check that calibration frames are specified
  if (!calibrationConfig->has("list_frames") || calibrationConfig->get("list_frames")->asList().empty()) {
    MsgBoxHandler::getInstance()->generic(tr("No calibration frames were selected or specified"), tr("Warning"),
                                          WARNING_ICON);
    return std::shared_ptr<VideoStitch::Ptv::Value>();
  }
  return calibrationConfig;
}

bool CalibrationWidget::addCustomRigToCalibrationConfig(std::shared_ptr<VideoStitch::Ptv::Value> calibrationConfig,
                                                        double fov,
                                                        VideoStitch::Core::InputDefinition::Format lensFormat) const {
  // FIXME TODOLATER fov == 0.0 is how the studio interface triggers the "auto_iterate_fov", a proper widget is needed
  // for it
  calibrationConfig->get("auto_iterate_fov")->asBool() = (fov <= 0.0);  // iterate over a predefined list of hfov values
  if (fov <= 0.0) {
    fov = PTV_DEFAULT_HFOV;
  }
  // end of FIXME TODOLATER
  calibrationConfig->get("initial_hfov")->asDouble() = fov;

  size_t nbCameras = panoDef->numVideoInputs();
  const VideoStitch::Core::InputDefinition& firstVideoInput = panoDef->getVideoInputs()[0];
  std::unique_ptr<VideoStitch::Core::RigDefinition> rigdef(VideoStitch::Core::RigDefinition::createBasicUnknownRig(
      "default", lensFormat, nbCameras, firstVideoInput.getWidth(), firstVideoInput.getHeight(),
      firstVideoInput.getCroppedWidth(), firstVideoInput.getCroppedHeight(), fov, panoDef.data()));

  if (rigdef == nullptr) {
    MsgBoxHandler::getInstance()->generic(tr("Invalid parameters for calibration"), tr("Warning"), WARNING_ICON);
    return false;
  }

  calibrationConfig->push("rig", rigdef->serialize());

  VideoStitch::Ptv::Value* listCameras = VideoStitch::Ptv::Value::emptyObject();
  for (auto it : rigdef->getRigCameraDefinitionMap()) {
    listCameras->asList().push_back(it.second->serialize());
  }
  calibrationConfig->push("cameras", listCameras);
  return true;
}

void CalibrationWidget::showCalibrationModes(bool show) {
  ui->radioAutomatic->setVisible(show);
  ui->radioImport->setVisible(show);
}

void CalibrationWidget::calibrationFrameSelectionModes(bool automatic) {
  ui->timeSequence->setVisible(automatic);
  ui->manualInputFramesWidget->setVisible(!automatic);
}

void CalibrationWidget::deshuffle() {
  VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Starting deshuffle algorithm" << std::endl;
  panoDef.reset(project->getPanoConst().get()->clone());

  // Run algorithm asynchronously
  algorithmType = AlgorithmType::Deshuffle;
  startComputationOf(std::bind(&CalibrationWidget::deshuffleComputation, this));
}

void CalibrationWidget::resetCalibration() {
  oldPanoDef.reset(project->getPanoConst().get()->clone());
  panoDef.reset(project->getPanoConst().get()->clone());
  panoDef->resetCalibration();
  /* The command takes the ownership of oldPanoDef and panoDef */
  CalibrationAppliedCommand* command = new CalibrationAppliedCommand(oldPanoDef.take(), panoDef.take(), this);
  qApp->findChild<QUndoStack*>()->push(command);
}

void CalibrationWidget::updateRigRelatedWidgets() {
  ui->deshuffleButton->setVisible(!ui->rigWidget->customRigIsSelected() &&
                                  VSSettings::getSettings()->getShowExperimentalFeatures());
}

void CalibrationWidget::applyRigPreset(const QString rig) {
  Q_UNUSED(rig);
  VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Starting applying rig preset" << std::endl;
  startComputation(int(CalibrationOption::ApplyPresetOnly));
}

void CalibrationWidget::adjustCrop() {
  CropWindow cropWindow(project, ui->rigWidget->getCurrentLensType(), 0, this);
  connect(&cropWindow.getCropWidget(), &CropWidget::reextract, this, &CalibrationWidget::reextract);
  connect(&cropWindow.getCropWidget(), &CropWidget::reqRegisterRender, this, &CalibrationWidget::reqRegisterRender);
  connect(&cropWindow.getCropWidget(), &CropWidget::reqUnregisterRender, this, &CalibrationWidget::reqUnregisterRender);
  connect(&cropWindow.getCropWidget(), &CropWidget::reqApplyCrops, this, &CalibrationWidget::reqApplyCrops);
  cropWindow.getCropWidget().initializeTabs();
  cropWindow.exec();
  cropWindow.getCropWidget().deinitializeTabs();
}

void CalibrationWidget::startComputation(int calibrationOptions) {
  panoDef.reset(project->getPanoConst().get()->clone());
  std::shared_ptr<VideoStitch::Ptv::Value> calibrationConfig = buildCalibrationConfig(calibrationOptions);
  if (!calibrationConfig) {
    panoDef.reset();
    return;
  }

  // Run algorithm asynchronously
  algorithmType = AlgorithmType::Calibration;
  startComputationOf(std::bind(&CalibrationWidget::calibrationComputation, this, calibrationConfig));
}

VideoStitch::Status* CalibrationWidget::calibrationComputation(
    std::shared_ptr<VideoStitch::Ptv::Value> calibrationConfig) {
  VideoStitch::Potential<VideoStitch::Util::Algorithm> fStatus =
      VideoStitch::Util::Algorithm::create("calibration", calibrationConfig.get());
  if (!fStatus.ok()) {
    panoDef.reset();
    return new VideoStitch::Status(VideoStitch::Origin::CalibrationAlgorithm, VideoStitch::ErrType::UnsupportedAction,
                                   tr("Could not initialize the calibration algorithm").toStdString(),
                                   fStatus.status());
  }
  algo.reset(fStatus.release());
  oldPanoDef.reset(project->getPanoConst().get()->clone());
  return new VideoStitch::Status(algo->apply(panoDef.data(), getReporter()).status());
}

VideoStitch::Status* CalibrationWidget::deshuffleComputation() {
  VideoStitch::Potential<VideoStitch::Util::Algorithm> fStatus = VideoStitch::Util::Algorithm::create("deshuffle_brt");
  if (!fStatus.ok()) {
    panoDef.reset();
    return new VideoStitch::Status(VideoStitch::Origin::CalibrationAlgorithm, VideoStitch::ErrType::UnsupportedAction,
                                   //: Error message
                                   tr("Could not initialize the deshuffle algorithm").toStdString(), fStatus.status());
  }
  algo.reset(fStatus.release());
  oldPanoDef.reset(project->getPanoConst().get()->clone());
  return new VideoStitch::Status(algo->apply(panoDef.data(), getReporter()).status());
}

QString CalibrationWidget::getAlgorithmName() const { return tr("Calibration"); }

void CalibrationWidget::manageComputationResult(bool hasBeenCancelled, VideoStitch::Status* status) {
  if (status->ok()) {
    VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Algorithm succeeded" << std::endl;
    if (algorithmType == AlgorithmType::Deshuffle) {
      PanoChangedCommand* command = new PanoChangedCommand(
          oldPanoDef.take(), panoDef.take(), QCoreApplication::translate("Undo command", "Deshuffle applied"));
      qApp->findChild<QUndoStack*>()->push(command);
    } else if (algorithmType == AlgorithmType::Calibration) {
      /* The command takes the ownership of oldPanoDef and panoDef */
      CalibrationAppliedCommand* command = new CalibrationAppliedCommand(oldPanoDef.take(), panoDef.take(), this);
      qApp->findChild<QUndoStack*>()->push(command);
    }
  } else {
    if (!hasBeenCancelled) {
      //: Algorithm error text
      const QString text =
          algorithmType == AlgorithmType::Deshuffle ? tr("Deshuffle failed") : tr("Automatic calibration failed");
      MsgBoxHandlerHelper::genericErrorMessage(
          {VideoStitch::Origin::CalibrationAlgorithm, VideoStitch::ErrType::RuntimeError, text.toStdString(), *status});
    }
    panoDef.reset();
    oldPanoDef.reset();
  }
  algorithmType = AlgorithmType::None;
  delete status;
}

void CalibrationWidget::refresh(mtime_t date) {
  VideoStitch::FrameRate frameRate = GlobalController::getInstance().getController()->getFrameRate();
  currentFrame = round((date / 1000000.0) * (frameRate.num / (double)frameRate.den));
}

void CalibrationWidget::addFrame(frameid_t frame) {
  int row = 0;
  for (; row < ui->listFrames->count(); ++row) {
    int64_t listItemFrame = ui->listFrames->item(row)->data(Qt::UserRole).toInt();
    if (listItemFrame == frame) {
      return;
    } else if (listItemFrame > frame) {
      break;
    }
  }

  QListWidgetItem* newFrameItem = new QListWidgetItem(ui->listFrames);
  newFrameItem->setText(tr("Frame %0").arg(frame));
  newFrameItem->setData(Qt::UserRole, QVariant(frame));
  ui->listFrames->insertItem(row, newFrameItem);
  ui->clearFramesButton->setEnabled(ui->listFrames->count() > 0);
}

void CalibrationWidget::addCurrentFrame() { addFrame(currentFrame); }

void CalibrationWidget::removeFrame() {
  if (!ui->listFrames->selectedItems().isEmpty()) {
    delete ui->listFrames->takeItem(ui->listFrames->row(ui->listFrames->selectedItems().first()));
  }
  ui->clearFramesButton->setEnabled(ui->listFrames->count() > 0);
}

void CalibrationWidget::clearFrames() {
  ui->listFrames->clear();
  ui->clearFramesButton->setEnabled(ui->listFrames->count() > 0);
}

void CalibrationWidget::seekFrame(QListWidgetItem* item) { emit reqSeek(frameid_t(item->data(Qt::UserRole).toInt())); }

void CalibrationWidget::generateSyntheticKeypointsStateChanged(int state) {
  // make sure checkBoxes in different windows have the same state
  ui->syntheticKeypointsCheckBox->blockSignals(true);
  ui->syntheticKeypointsCheckBox_InImprove->blockSignals(true);
  ui->syntheticKeypointsCheckBox->setCheckState(Qt::CheckState(state));
  ui->syntheticKeypointsCheckBox_InImprove->setCheckState(Qt::CheckState(state));
  ui->syntheticKeypointsCheckBox->blockSignals(false);
  ui->syntheticKeypointsCheckBox_InImprove->blockSignals(false);
}

void CalibrationWidget::calibrationModeChanged(bool automatic) {
  if (automatic) {
    ui->stackedWidget->setCurrentWidget(ui->pageAutomatic);
  } else {
    ui->stackedWidget->setCurrentWidget(ui->pageImport);
  }
}

void CalibrationWidget::onProjectOpened(ProjectDefinition* project) {
  if (project != this->project) {
    ui->radioFrameAutomatic->setChecked(true);
    ui->radioAutomatic->setChecked(true);
    clearFrames();
  }
  ComputationWidget::onProjectOpened(project);
  ui->rigWidget->setProject(project);
  updateRigValues();
}

void CalibrationWidget::clearProject() {
  ComputationWidget::clearProject();
  ui->automaticCalibStackedWidget->setCurrentWidget(ui->pageParameters);
  ui->rigWidget->clearProject();
  ui->radioFrameAutomatic->setChecked(true);
  clearFrames();
  showCalibrationModes(true);
}

void CalibrationWidget::frameItemSelected() {
  ui->removeFrameButton->setEnabled(!ui->listFrames->selectedItems().isEmpty());
}

void CalibrationWidget::updateRigValues() { updateWhenCalibrationChanged(project->getPanoConst().get()); }

void CalibrationWidget::updateSequence(const QString start, const QString stop) {
  ui->timeSequence->sequenceUpdated(start, stop);
}
