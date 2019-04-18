// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "blendingmaskwidget.hpp"
#include "ui_blendingmaskwidget.h"

#include "videostitcher/postprodprojectdefinition.hpp"
#include "videostitcher/globalpostprodcontroller.hpp"

#include "libvideostitch-gui/dialogs/modalprogressdialog.hpp"
#include "libvideostitch-gui/videostitcher/presetsmanager.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/LibLogHelpers.hpp"
#include "libvideostitch-gui/mainwindow/ui_header/progressreporterwrapper.hpp"
#include "libvideostitch/parse.hpp"
#include "commands/blendingmaskappliedcommand.hpp"

#include <memory>

BlendingMaskWidget::BlendingMaskWidget(QWidget* parent)
    : ComputationWidget(parent),
      panoDef(nullptr),
      oldPanoDef(nullptr),
      currentFrame(-1),
      ui(new Ui::BlendingMaskWidget) {
  ui->setupUi(this);
  connect(ui->addBlendingMaskButton, &QPushButton::clicked, this, &BlendingMaskWidget::startBlendingMaskComputation);
  connect(ui->checkEnableBlendingMask, &QCheckBox::toggled, this, &BlendingMaskWidget::onEnableBlendingMaskChecked);
  connect(ui->checkEnableInterpolation, &QCheckBox::toggled, this, &BlendingMaskWidget::onEnableInterpolationChecked);
  connect(ui->listFrames, &QListWidget::itemDoubleClicked, this, &BlendingMaskWidget::seekFrame);
  connect(ui->listFrames, &QListWidget::itemClicked, this, &BlendingMaskWidget::frameItemSelected);
  connect(ui->removeFrameButton, &QPushButton::clicked, this, &BlendingMaskWidget::removeFrame);
  connect(ui->clearFramesButton, &QPushButton::clicked, this, &BlendingMaskWidget::clearFrames);
}

BlendingMaskWidget::~BlendingMaskWidget() { delete ui; }

void BlendingMaskWidget::seekFrame(QListWidgetItem* item) { emit reqSeek(frameid_t(item->data(Qt::UserRole).toInt())); }

void BlendingMaskWidget::setFrame(frameid_t frame) {
  currentFrame = frame;
  setFrameText();
}

void BlendingMaskWidget::removeFrame() {
  if (!ui->listFrames->selectedItems().isEmpty()) {
    delete ui->listFrames->takeItem(ui->listFrames->row(ui->listFrames->selectedItems().first()));
  }
  ui->clearFramesButton->setEnabled(ui->listFrames->count() > 0);
  ui->removeFrameButton->setEnabled(false);
}

void BlendingMaskWidget::clearListFrame() {
  ui->listFrames->clear();
  ui->clearFramesButton->setEnabled(ui->listFrames->count() > 0);
  ui->removeFrameButton->setEnabled(false);
}

void BlendingMaskWidget::clearFrames() {
  clearListFrame();
  oldPanoDef = project->getPanoConst().get()->clone();
  panoDef = project->getPanoConst().get()->clone();
  const std::vector<frameid_t> frameIds = project->getPanoConst().get()->getBlendingMaskFrameIds();
  panoDef->removeBlendingMaskFrameIds(frameIds);
  BlendingMaskAppliedCommand* command = new BlendingMaskAppliedCommand(oldPanoDef, panoDef, this);
  qApp->findChild<QUndoStack*>()->push(command);
  panoDef = nullptr;
  oldPanoDef = nullptr;
}

void BlendingMaskWidget::frameItemSelected() { ui->removeFrameButton->setEnabled(true); }

void BlendingMaskWidget::addFrame(const frameid_t frame) {
  std::vector<frameid_t> frames;
  for (int row = 0; row < ui->listFrames->count(); row++) {
    int64_t listItemFrame = ui->listFrames->item(row)->data(Qt::UserRole).toInt();
    if (listItemFrame == frame) {
      return;
    }
    frames.push_back((frameid_t)listItemFrame);
  }
  frames.push_back(frame);
  std::sort(frames.begin(), frames.end(), std::less<frameid_t>());

  ui->listFrames->clear();
  for (size_t row = 0; row < frames.size(); row++) {
    QListWidgetItem* newFrameItem = new QListWidgetItem(ui->listFrames);
    newFrameItem->setText(QString(tr("Frame %0")).arg(frames[row]));
    newFrameItem->setData(Qt::UserRole, QVariant(frames[row]));
    ui->listFrames->insertItem((int)row, newFrameItem);
  }

  ui->clearFramesButton->setEnabled(ui->listFrames->count() > 0);
}

void BlendingMaskWidget::onEnableBlendingMaskChecked(const bool show) {
  if (!project) {
    return;
  }
  panoDef = project->getPanoConst().get()->clone();
  panoDef->setBlendingMaskEnabled(show);
  ui->addBlendingMaskButton->setEnabled(show);
  emit reqApplyBlendingMask(panoDef);
  panoDef = nullptr;
}

void BlendingMaskWidget::onEnableInterpolationChecked(const bool show) {
  if (!project) {
    return;
  }
  panoDef = project->getPanoConst().get()->clone();
  panoDef->setBlendingMaskInterpolationEnabled(show);
  emit reqApplyBlendingMask(panoDef);
  panoDef = nullptr;
}

void BlendingMaskWidget::setFrameText() {
  if (!project) {
    return;
  }

  const std::pair<frameid_t, frameid_t> boundedFrames =
      project->getPanoConst().get()->getBlendingMaskBoundedFrameIds(currentFrame);
  const frameid_t prevFrame = (frameid_t)boundedFrames.first;
  const frameid_t nextFrame = (frameid_t)boundedFrames.second;

  if ((prevFrame == nextFrame) || (prevFrame < currentFrame && nextFrame < currentFrame)) {
    // This is the exact frame, just take the computed mask
    ui->checkEnableBlendingMask->setText(
        QString(tr("Enable blending mask computed at frame (%1)")).arg(QString::number(prevFrame)));
  } else if (prevFrame > currentFrame && nextFrame > currentFrame) {
    // Use the first computed frame
    ui->checkEnableBlendingMask->setText(
        QString(tr("Enable blending mask computed at frame (%1)")).arg(QString::number(nextFrame)));
  } else if (prevFrame < currentFrame && nextFrame > currentFrame) {
    // Current frame is in the middle of two computed frames. Interpolate the in-between result
    ui->checkEnableBlendingMask->setText(QString(tr("Enable interpolated blending mask between frame (%1) and (%2)"))
                                             .arg(QString::number(prevFrame), QString::number(nextFrame)));
  }
}

void BlendingMaskWidget::onProjectOpened(ProjectDefinition* p) {
  ComputationWidget::onProjectOpened(p);
  if (!project) {
    return;
  }
  const bool blendingMaskEnabled = project->getPanoConst().get()->getBlendingMaskEnabled();
  const bool interpolationEnabled = project->getPanoConst().get()->getBlendingMaskInterpolationEnabled();
  clearListFrame();
  const std::vector<frameid_t> frameIds = project->getPanoConst().get()->getBlendingMaskFrameIds();
  for (auto frameId : frameIds) {
    addFrame(frameId);
  }
  ui->addBlendingMaskButton->setEnabled(blendingMaskEnabled);
  ui->checkEnableBlendingMask->setChecked(blendingMaskEnabled);
  ui->checkEnableInterpolation->setChecked(interpolationEnabled);
  setFrameText();
}

VideoStitch::Status* BlendingMaskWidget::computeBlendingMask(
    std::shared_ptr<VideoStitch::Ptv::Value> blendingMaskConfig) {
  VideoStitch::Potential<VideoStitch::Util::Algorithm> fStatus =
      VideoStitch::Util::Algorithm::create("mask", blendingMaskConfig.get());
  if (!fStatus.ok()) {
    MsgBoxHandler::getInstance()->generic(tr("Blending mask error"), tr("Could not create the blending mask algorithm"),
                                          CRITICAL_ERROR_ICON);
    delete panoDef;
    panoDef = nullptr;
    return new VideoStitch::Status(fStatus.status());
  }
  algo.reset(fStatus.release());
  oldPanoDef = project->getPanoConst().get()->clone();
  panoDef = project->getPanoConst().get()->clone();
  return new VideoStitch::Status(algo->apply(panoDef, getReporter()).status());
}

void BlendingMaskWidget::startBlendingMaskComputation() {
  std::shared_ptr<VideoStitch::Ptv::Value> blendingMaskConfig(
      PresetsManager::getInstance()->clonePresetContent("mask", "default_studio").release());
  if (blendingMaskConfig == nullptr) {
    MsgBoxHandler::getInstance()->generic(tr("Could not initialize the calibration"), tr("Warning"), WARNING_ICON);
    return;
  }
  if (currentFrame >= 0) {
    VideoStitch::Ptv::Value* list_frames = VideoStitch::Ptv::Value::emptyObject();
    VideoStitch::Ptv::Value* val = VideoStitch::Ptv::Value::emptyObject();
    val->asInt() = currentFrame;
    list_frames->asList().push_back(val);
    blendingMaskConfig->push("list_frames", list_frames);
  }
  startComputationOf(std::bind(&BlendingMaskWidget::computeBlendingMask, this,
                               std::shared_ptr<VideoStitch::Ptv::Value>(blendingMaskConfig)));
}

void BlendingMaskWidget::refresh(mtime_t date) {
  VideoStitch::FrameRate frameRate = GlobalController::getInstance().getController()->getFrameRate();
  currentFrame = round((date / 1000000.0) * (frameRate.num / (double)frameRate.den));
  ui->addBlendingMaskButton->setText(QString(tr("Add blending mask at frame (%0)")).arg(currentFrame));
  setFrameText();
}

QString BlendingMaskWidget::getAlgorithmName() const { return tr("Blending mask"); }

void BlendingMaskWidget::manageComputationResult(bool hasBeenCancelled, VideoStitch::Status* status) {
  if (status->ok()) {
    /* Copy from constant to result*/
    /* The command takes the ownership of oldPanoDef and panoDef */
    BlendingMaskAppliedCommand* command = new BlendingMaskAppliedCommand(oldPanoDef, panoDef, this);
    qApp->findChild<QUndoStack*>()->push(command);
    setFrameText();
    addFrame(currentFrame);
  } else {
    if (!hasBeenCancelled) {
      MsgBoxHandlerHelper::genericErrorMessage({VideoStitch::Origin::BlendingMaskAlgorithm,
                                                VideoStitch::ErrType::RuntimeError,
                                                tr("Blending mask optimization failed").toStdString(), *status});
    }
    delete panoDef;
    delete oldPanoDef;
  }
  panoDef = nullptr;
  oldPanoDef = nullptr;
  delete status;
}
