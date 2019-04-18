// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "stabilizationwidget.hpp"
#include "ui_stabilizationwidget.h"

#include "commands/stabilizationcomputedcommand.hpp"
#include "videostitcher/postprodprojectdefinition.hpp"
#include "videostitcher/globalpostprodcontroller.hpp"

#include "libvideostitch-gui/mainwindow/LibLogHelpers.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/ui_header/progressreporterwrapper.hpp"
#include "libvideostitch-gui/utils/notonlyvideosfilterer.hpp"

StabilizationWidget::StabilizationWidget(QWidget* parent)
    : ComputationWidget(parent), ui(new Ui::StabilizationWidget), panoDef(nullptr), oldPanoDef(nullptr), ctx(nullptr) {
  ui->setupUi(this);
  ui->orientationWarningLabel->setVisible(false);
  ui->labelWarningIcon->setVisible(false);
  ui->editOrientationButton->setEnabled(false);
  NotOnlyVideosFilterer::getInstance()->watch(ui->stabilizationGroupBox);

  connect(ui->computeOnRangeButton, &QPushButton::clicked, this, &StabilizationWidget::startComputation);
  connect(ui->resetButton, &QPushButton::clicked, this, &StabilizationWidget::onResetStabilizationClicked);
  connect(ui->editOrientationButton, &QPushButton::toggled, this, &StabilizationWidget::onButtonOrientationToggled);
  connect(ui->editOrientationButton, &QPushButton::toggled, this, &StabilizationWidget::orientationActivated);
}

StabilizationWidget::~StabilizationWidget() { delete ctx; }

void StabilizationWidget::startComputation() {
  std::shared_ptr<VideoStitch::Ptv::Value> stabilizationConfig(VideoStitch::Ptv::Value::emptyObject());
  stabilizationConfig->get("first_frame")->asInt() = project->getFirstFrame();
  stabilizationConfig->get("last_frame")->asInt() = project->getLastFrame();
  VideoStitch::FrameRate frameRate = GlobalController::getInstance().getController()->getFrameRate();
  stabilizationConfig->get("convolution_span")->asInt() =
      2 * frameRate.num / frameRate.den;  // empirical number, correct values between 1 sec and 3 sec

  startComputationOf(std::bind(&StabilizationWidget::computation, this, stabilizationConfig));
}

VideoStitch::Status* StabilizationWidget::computation(std::shared_ptr<VideoStitch::Ptv::Value> stabilizationConfig) {
  VideoStitch::Potential<VideoStitch::Util::Algorithm> fStatus =
      VideoStitch::Util::Algorithm::create("stabilization", stabilizationConfig.get());
  if (!fStatus.ok()) {
    delete panoDef;
    panoDef = NULL;
    return new VideoStitch::Status(VideoStitch::Origin::StabilizationAlgorithm, VideoStitch::ErrType::UnsupportedAction,
                                   tr("Could not initialize the stabilization algorithm").toStdString(),
                                   fStatus.status());
  }
  algo.reset(fStatus.release());
  panoDef = project->getPanoConst().get()->clone();
  oldPanoDef = project->getPanoConst().get()->clone();
  return new VideoStitch::Status(algo->apply(panoDef, getReporter(), &ctx).status());
}

void StabilizationWidget::onProjectOpened(ProjectDefinition* p) {
  if (project) {
    disconnect(this, &StabilizationWidget::reqSetEditOrientationActivated, project,
               &ProjectDefinition::setDisplayOrientationGrid);
  }

  ComputationWidget::onProjectOpened(p);
  ui->editOrientationButton->setChecked(project->getDisplayOrientationGrid());

  connect(this, SIGNAL(reqResetStabilization()), p, SLOT(resetStabilizationCurve()), Qt::UniqueConnection);
  connect(this, &StabilizationWidget::reqSetEditOrientationActivated, project,
          &ProjectDefinition::setDisplayOrientationGrid, Qt::UniqueConnection);
}

void StabilizationWidget::clearProject() {
  if (project) {
    disconnect(this, &StabilizationWidget::reqSetEditOrientationActivated, project,
               &ProjectDefinition::setDisplayOrientationGrid);
  }
  ComputationWidget::clearProject();

  delete ctx;
  ctx = nullptr;
  ui->editOrientationButton->setChecked(false);
}

void StabilizationWidget::onResetStabilizationClicked() { emit reqResetStabilization(); }

void StabilizationWidget::onProjectOrientable(bool yprModificationsAllowed) {
  //: Warning displayed when the orientation button is disabled
  ui->orientationWarningLabel->setText(
      tr("Orientation cannot be adjusted when using %0 type of projection").arg(project->getProjection()));
  ui->orientationWarningLabel->setVisible(!yprModificationsAllowed);
  ui->labelWarningIcon->setVisible(!yprModificationsAllowed);
  ui->editOrientationButton->setEnabled(yprModificationsAllowed);
}

void StabilizationWidget::toggleOrientationButton() {
  if (project && ui->editOrientationButton->isEnabled()) {
    ui->editOrientationButton->toggle();
  }
}

QString StabilizationWidget::getAlgorithmName() const { return tr("Stabilization"); }

void StabilizationWidget::manageComputationResult(bool hasBeenCancelled, VideoStitch::Status* status) {
  if (status->ok()) {
    StabilizationComputedCommand* command = new StabilizationComputedCommand(oldPanoDef, panoDef, this);
    qApp->findChild<QUndoStack*>()->push(command);
    panoDef = nullptr;
    oldPanoDef = nullptr;
  } else {
    if (!hasBeenCancelled) {
      MsgBoxHandlerHelper::genericErrorMessage({VideoStitch::Origin::StabilizationAlgorithm,
                                                VideoStitch::ErrType::RuntimeError,
                                                tr("Stabilization failed").toStdString(), *status});
    }
  }
  delete status;
}

void StabilizationWidget::reset() {
  ui->editOrientationButton->setChecked(false);
  onButtonOrientationToggled(false);
}

void StabilizationWidget::onButtonOrientationToggled(const bool activate) {
  emit reqSetEditOrientationActivated(activate, true);
}

void StabilizationWidget::updateSequence(const QString start, const QString stop) {
  ui->timeSequence->sequenceUpdated(start, stop);
}
