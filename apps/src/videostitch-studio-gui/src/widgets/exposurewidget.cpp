// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exposurewidget.hpp"
#include "ui_exposurewidget.h"

#include "../videostitcher/postprodprojectdefinition.hpp"
#include "videostitcher/globalpostprodcontroller.hpp"

#include "libvideostitch-gui/mainwindow/LibLogHelpers.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/ui_header/progressreporterwrapper.hpp"
#include "libvideostitch-gui/utils/imagesorproceduralsonlyfilterer.hpp"

#include "libvideostitch-base/file.hpp"
#include "libvideostitch-base/vslog.hpp"

#include "commands/exposureappliedcommand.hpp"
#include "commands/photometriccalibrationappliedcommand.hpp"

ExposureWidget::ExposureWidget(QWidget* parent)
    : ComputationWidget(parent), ui(new Ui::ExposureWidget), algorithmType(AlgorithmType::None), currentFrame(-1) {
  ui->setupUi(this);
  ui->photometricWidget->hide();  // Hack because the photometric widget has a big minimum size
  ui->photometryStackedWidget->setCurrentWidget(ui->photometryParametersPage);
  ui->whiteBalanceRadioButton->setChecked(true);
  ui->advancedExpoParametersBox->setChecked(false);
  ui->exposureRadioButton->setAttribute(Qt::WA_LayoutUsesWidgetRect);
  ui->whiteBalanceRadioButton->setAttribute(Qt::WA_LayoutUsesWidgetRect);

  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->frameStepLabel, FeatureFilterer::PropertyToWatch::visible);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->frameStepBox, FeatureFilterer::PropertyToWatch::visible);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->compensateButton);

  connect(ui->calibratePhotometricButton, &QPushButton::clicked, this, &ExposureWidget::startPhotometricCalibration);
  connect(ui->resetPhotometryButton, &QPushButton::clicked, this, &ExposureWidget::reqResetPhotometricCalibration);
  connect(ui->compensateButton, &QPushButton::clicked, this, &ExposureWidget::startExposureCompensationOnSequence);
  connect(ui->currentFrameButton, &QPushButton::clicked, this, &ExposureWidget::startExposureCompensationHere);
  connect(ui->clearSequenceButton, &QPushButton::clicked, this, &ExposureWidget::resetExposureCompensationOnSequence);
}

ExposureWidget::~ExposureWidget() {}

void ExposureWidget::resetEvCurves() { emit reqResetEvCurves(true); }

void ExposureWidget::onProjectOpened(ProjectDefinition* p) {
  ComputationWidget::onProjectOpened(p);
  connect(this, SIGNAL(reqResetEvCurves(bool)), project, SLOT(resetEvCurves(bool)), Qt::UniqueConnection);
  connect(this, SIGNAL(reqResetEvCurvesSequence(frameid_t, frameid_t)), project,
          SLOT(resetEvCurvesSequence(frameid_t, frameid_t)), Qt::UniqueConnection);
  connect(this, &ExposureWidget::reqResetPhotometricCalibration, project,
          &PostProdProjectDefinition::resetPhotometricCalibration, Qt::UniqueConnection);
  connect(project, &PostProdProjectDefinition::reqRefreshPhotometry, this, &ExposureWidget::updatePhotometryResults,
          Qt::UniqueConnection);
  updatePhotometryResults();

  VideoStitch::FrameRate frameRate = GlobalController::getInstance().getController()->getFrameRate();
  ui->frameStepBox->setValue(2 * std::ceil(frameRate.num / frameRate.den));

  QStringList prefixedNames;
  //: Exposure compensation anchor
  prefixedNames << tr("All");
  auto nbInputs = project->getNumInputs();
  for (readerid_t index = 0; index < nbInputs; ++index) {
    const VideoStitch::Core::InputDefinition& inputDef = project->getPanoConst()->getInput(index);
    QString name =
        QString("%0 - %1").arg(index).arg(File::strippedName(QString::fromStdString(inputDef.getDisplayName())));
    prefixedNames << name;
  }
  ui->anchorComboBox->clear();
  ui->anchorComboBox->addItems(prefixedNames);
}

void ExposureWidget::clearProject() {
  ComputationWidget::clearProject();
  ui->anchorComboBox->clear();
  ui->photometryStackedWidget->setCurrentWidget(ui->photometryParametersPage);
}

void ExposureWidget::updateSequence(const QString start, const QString stop) {
  ui->timeSequence->sequenceUpdated(start, stop);
}

void ExposureWidget::updateToPano(frameid_t newLastStitchableFrame, frameid_t newCurrentFrame) {
  Q_UNUSED(newCurrentFrame);
  ui->frameStepBox->setRange(1, newLastStitchableFrame);
}

VideoStitch::Status* ExposureWidget::computeExposureCompensation(std::shared_ptr<VideoStitch::Ptv::Value> config) {
  VideoStitch::Potential<VideoStitch::Util::Algorithm> fStatus =
      VideoStitch::Util::Algorithm::create("exposure_stabilize", config.get());
  if (!fStatus.ok()) {
    panoDef.reset();
    return new VideoStitch::Status(VideoStitch::Origin::ExposureAlgorithm, VideoStitch::ErrType::UnsupportedAction,
                                   tr("Could not create the exposure stabilization algorithm").toStdString(),
                                   fStatus.status());
  }
  algo.reset(fStatus.release());
  panoDef.reset(project->getPanoConst().get()->clone());
  oldPanoDef.reset(project->getPanoConst().get()->clone());
  return new VideoStitch::Status(algo->apply(panoDef.data(), getReporter()).status());
}

void ExposureWidget::refresh(mtime_t date) {
  VideoStitch::FrameRate frameRate = GlobalController::getInstance().getController()->getFrameRate();
  currentFrame = round((date / 1000000.0) * (frameRate.num / (double)frameRate.den));
}

void ExposureWidget::startExposureCompensationOnSequence() { startExposureCompensation(false); }

void ExposureWidget::startExposureCompensationHere() { startExposureCompensation(true); }

void ExposureWidget::startPhotometricCalibration() {
  std::shared_ptr<VideoStitch::Ptv::Value> config(VideoStitch::Ptv::Value::emptyObject());
  config->get("first_frame")->asInt() = project->getFirstFrame();
  config->get("last_frame")->asInt() = project->getLastFrame();

  panoDef.reset(project->getPanoConst().get()->clone());
  algorithmType = AlgorithmType::PhotometricCalibration;
  startComputationOf(std::bind(&ExposureWidget::computePhotometricCalibration, this, config));
}

void ExposureWidget::updatePhotometryResults() {
  ui->photometricWidget->setVisible(
      project->getPanoConst()
          ->photometryHasBeenCalibrated());  // Hack because the photometric widget has a big minimum size
  ui->photometryStackedWidget->setCurrentWidget(project->getPanoConst()->photometryHasBeenCalibrated()
                                                    ? ui->photometryResultsPage
                                                    : ui->photometryParametersPage);

  if (project->getPanoConst()->numVideoInputs()) {
    const VideoStitch::Core::InputDefinition& firstVideoInput = project->getPanoConst()->getVideoInputs()[0];
    ui->photometricWidget->setVignetteValues(firstVideoInput.getVignettingCoeff1(),
                                             firstVideoInput.getVignettingCoeff2(),
                                             firstVideoInput.getVignettingCoeff3());
    ui->photometricWidget->setEmorValues(firstVideoInput.getEmorA(), firstVideoInput.getEmorB(),
                                         firstVideoInput.getEmorC(), firstVideoInput.getEmorD(),
                                         firstVideoInput.getEmorE());
  }
}

QString ExposureWidget::getAlgorithmName() const {
  //: Color correction progress dialog title
  return tr("Color correction");
}

void ExposureWidget::manageComputationResult(bool hasBeenCancelled, VideoStitch::Status* status) {
  if (status->ok()) {
    if (algorithmType == AlgorithmType::ExposureCompensation) {
      ExposureAppliedCommand* command = new ExposureAppliedCommand(oldPanoDef.take(), panoDef.take(), this);
      qApp->findChild<QUndoStack*>()->push(command);
    } else if (algorithmType == AlgorithmType::PhotometricCalibration) {
      PhotometricCalibrationAppliedCommand* command =
          new PhotometricCalibrationAppliedCommand(oldPanoDef.take(), panoDef.take(), this);
      qApp->findChild<QUndoStack*>()->push(command);
    }
  } else {
    if (!hasBeenCancelled) {
      MsgBoxHandlerHelper::genericErrorMessage({VideoStitch::Origin::ExposureAlgorithm,
                                                VideoStitch::ErrType::RuntimeError,
                                                tr("Exposure adjustment failed").toStdString(), *status});
    }
    panoDef.reset();
    oldPanoDef.reset();
  }

  algorithmType = AlgorithmType::None;
  delete status;
}

void ExposureWidget::startExposureCompensation(bool computeCurrentFrame) {
  frameid_t firstFrame = computeCurrentFrame ? currentFrame : project->getFirstFrame();
  // FIXME : To create a KF at the current frame : If the first and last frame are the same, the algo doesn't compute.
  // If there is one frame of difference, it works and the KF is created at the first KF. That is why we add +1 to
  // lastframe
  frameid_t lastFrame = computeCurrentFrame ? (currentFrame + 1) : project->getLastFrame();
  std::shared_ptr<VideoStitch::Ptv::Value> expoConfig(VideoStitch::Ptv::Value::emptyObject());
  expoConfig->get("first_frame")->asInt() = firstFrame;
  expoConfig->get("last_frame")->asInt() = lastFrame;
  expoConfig->get("anchor_keyframes")->asBool() = true;
  expoConfig->get("time_step")->asInt() =
      std::min(project->getLastFrame() - project->getFirstFrame(), ui->frameStepBox->value());
  expoConfig->get("stabilize_wb")->asBool() = ui->whiteBalanceRadioButton->isChecked();
  expoConfig->get("anchor")->asInt() =
      (ui->anchorComboBox->currentIndex() < 0) ? -1 : ui->anchorComboBox->currentIndex() - 1;
  expoConfig->get("keep_origin_if_constant")->asBool() = false;

  panoDef.reset(project->getPanoConst().get()->clone());
  algorithmType = AlgorithmType::ExposureCompensation;
  startComputationOf(std::bind(&ExposureWidget::computeExposureCompensation, this, expoConfig));
}

VideoStitch::Status* ExposureWidget::computePhotometricCalibration(std::shared_ptr<VideoStitch::Ptv::Value> config) {
  VideoStitch::Potential<VideoStitch::Util::Algorithm> fStatus =
      VideoStitch::Util::Algorithm::create("photometric_calibration", config.get());
  if (!fStatus.ok()) {
    MsgBoxHandler::getInstance()->generic(
        tr("Vignetting error"), tr("Could not create the photometric calibration algorithm"), CRITICAL_ERROR_ICON);
    panoDef.reset();
    return new VideoStitch::Status(fStatus.status());
  }
  algo.reset(fStatus.release());
  panoDef.reset(project->getPanoConst().get()->clone());
  oldPanoDef.reset(project->getPanoConst().get()->clone());
  return new VideoStitch::Status(algo->apply(panoDef.data(), getReporter()).status());
}

void ExposureWidget::resetExposureCompensationOnSequence() {
  emit reqResetEvCurvesSequence(project->getFirstFrame(), project->getLastFrame());
}
