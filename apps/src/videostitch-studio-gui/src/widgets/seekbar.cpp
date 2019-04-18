// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "seekbar.hpp"
#include "ui_seekbar.h"

#include "commands/workingareachangedcommand.hpp"
#include "timeline/timelineticks.hpp"
#include "videostitcher/postprodprojectdefinition.hpp"

#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"
#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"
#include "libvideostitch-gui/mainwindow/objectutil.hpp"
#include "libvideostitch-gui/mainwindow/statemanager.hpp"
#include "libvideostitch-gui/mainwindow/timeconverter.hpp"
#include "libvideostitch-gui/utils/imagesorproceduralsonlyfilterer.hpp"

SeekBar::SeekBar(QWidget *parent)
    : QWidget(parent), ui(new Ui::SeekBar), comp(SignalCompressionCaps::createOwned()), project(nullptr) {
  ui->setupUi(this);

  // misc
  pause();
  setEnabled(false);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->playButton);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->toStartButton);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->toStopButton);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->extendedTimelineContainer);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->currentFrameWidget,
                                                        FeatureFilterer::PropertyToWatch::visible);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->firstFrameWidget,
                                                        FeatureFilterer::PropertyToWatch::visible);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->lastFrameWidget, FeatureFilterer::PropertyToWatch::visible);
  ui->firstFrameWidget->setEditable(true);
  ui->firstFrameWidget->setFramenumberVisibility(false);
  ui->lastFrameWidget->setEditable(true);
  ui->lastFrameWidget->setFramenumberVisibility(false);
  connect(ui->firstFrameWidget, SIGNAL(frameChangedFromTimeCode(frameid_t)), this, SLOT(startTimeEdited(frameid_t)));
  connect(ui->lastFrameWidget, SIGNAL(frameChangedFromTimeCode(frameid_t)), this, SLOT(stopTimeEdited(frameid_t)));

  ui->curvesTree->setTimeline(ui->extendedTimelineContainer->getTimeline());
  StateManager::getInstance()->registerObject(this);
}

SeekBar::~SeekBar() { delete ui; }

void SeekBar::setTimeline() {
  ui->extendedTimelineContainer->setVisible(true);
  ui->curvesTree->setVisible(true);

  VideoStitch::Helper::toggleConnect(true, this, SIGNAL(reqAddKeyframe()), ui->extendedTimelineContainer->getTimeline(),
                                     SLOT(addKeyframeHere()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(true, this, SIGNAL(reqRefreshCurves()),
                                     ui->extendedTimelineContainer->getTimeline(), SIGNAL(reqRefreshCurves()),
                                     Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(true, ui->extendedTimelineContainer->getTimeline(), SIGNAL(reqSeek(frameid_t)),
                                     this, SLOT(seek(frameid_t)), Qt::UniqueConnection);

  VideoStitch::Helper::toggleConnect(
      true, ui->extendedTimelineContainer->getTimeline(),
      SIGNAL(reqCurveChanged(SignalCompressionCaps *, VideoStitch::Core::Curve *, CurveGraphicsItem::Type, int)), this,
      SIGNAL(reqCurveChanged(SignalCompressionCaps *, VideoStitch::Core::Curve *, CurveGraphicsItem::Type, int)),
      Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(
      true, ui->extendedTimelineContainer->getTimeline(),
      SIGNAL(reqQuaternionCurveChanged(SignalCompressionCaps *, VideoStitch::Core::QuaternionCurve *,
                                       CurveGraphicsItem::Type, int)),
      this,
      SIGNAL(reqQuaternionCurveChanged(SignalCompressionCaps *, VideoStitch::Core::QuaternionCurve *,
                                       CurveGraphicsItem::Type, int)),
      Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(true, ui->extendedTimelineContainer->getTimeline(), SIGNAL(reqResetCurves()), this,
                                     SIGNAL(reqResetCurves()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(true, ui->curvesTree, SIGNAL(reqResetCurve(CurveGraphicsItem::Type, int)), this,
                                     SIGNAL(reqResetCurve(CurveGraphicsItem::Type, int)), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(true, ui->extendedTimelineContainer->getTimeline(),
                                     SIGNAL(reqResetCurve(CurveGraphicsItem::Type, int)), this,
                                     SIGNAL(reqResetCurve(CurveGraphicsItem::Type, int)), Qt::UniqueConnection);
  connect(ui->extendedTimelineContainer->getTimelineTicks(), &TimelineTicks::lowerBoundHandlePositionChanged, this,
          &SeekBar::startTimeEdited, Qt::UniqueConnection);
  connect(ui->extendedTimelineContainer->getTimelineTicks(), &TimelineTicks::upperBoundHandlePositionChanged, this,
          &SeekBar::stopTimeEdited, Qt::UniqueConnection);
  enableInternal();
}

CurvesTreeWidget *SeekBar::getCurvesTreeWidget() { return ui->curvesTree; }

void SeekBar::reset() {
  ui->extendedTimelineContainer->getTimeline()->resetTimeline();
  ui->curvesTree->removePreviouslySelectedFromTimeline();
}

void SeekBar::setValue(int val) { ui->extendedTimelineContainer->getTimeline()->moveCursorTo(val); }

void SeekBar::setCurrentFrameLabel(int frame) { ui->currentFrameWidget->setFrame(frame); }

void SeekBar::updateSequence() {
  VideoStitch::FrameRate frameRate = GlobalController::getInstance().getController()->getFrameRate();
  const QString start = TimeConverter::frameToTimeDisplay(project->getFirstFrame(), frameRate);
  const QString stop = TimeConverter::frameToTimeDisplay(project->getLastFrame(), frameRate);
  emit reqUpdateSequence(start, stop);
}

void SeekBar::updateCurves() { ui->curvesTree->populate(*project); }

frameid_t SeekBar::getCurrentFrameFromCursorPosition() const {
  return ui->extendedTimelineContainer->getTimeline()->value();
}

frameid_t SeekBar::getMinimumFrame() const { return ui->extendedTimelineContainer->getTimeline()->minimum(); }

frameid_t SeekBar::getMaximumFrame() const { return ui->extendedTimelineContainer->getTimeline()->maximum(); }

void SeekBar::changeState(GUIStateCaps::State s) {
  switch (s) {
    case GUIStateCaps::disabled:
    case GUIStateCaps::frozen:
      state = s;
      setEnabled(false);
      ui->extendedTimelineContainer->setDisabled(true);
      break;
    case GUIStateCaps::idle:
      state = s;
      ui->extendedTimelineContainer->getTimeline()->setValue(0);
      ui->firstFrameWidget->reset();
      ui->lastFrameWidget->reset();
      ui->currentFrameWidget->reset();
      ui->extendedTimelineContainer->setDisabled(true);
      setEnabled(false);
      break;
    case GUIStateCaps::stitch:
      ui->extendedTimelineContainer->setEnabled(true);
      state = s;
      setEnabled(true);
      break;
    default:
      Q_ASSERT(0);
      return;
  }
}

void SeekBar::setProject(ProjectDefinition *p) {
  project = qobject_cast<PostProdProjectDefinition *>(p);
  ui->extendedTimelineContainer->getTimeline()->setProject(p);
  ui->extendedTimelineContainer->getTimelineTicks()->setProject(project);
  connect(project, SIGNAL(reqRefreshCurves()), this, SIGNAL(reqRefreshCurves()), Qt::UniqueConnection);
  connect(project, SIGNAL(reqSetWorkingArea(frameid_t, frameid_t)), this, SLOT(setWorkingArea(frameid_t, frameid_t)),
          Qt::UniqueConnection);
  connect(project, &PostProdProjectDefinition::imagesOrProceduralsOnlyHasChanged, this, &SeekBar::updateCurves,
          Qt::UniqueConnection);
  initialize();
}

void SeekBar::clearProject() {
  project = nullptr;
  ui->extendedTimelineContainer->getTimeline()->setProject(nullptr);
  ui->extendedTimelineContainer->getTimeline()->setBounds(0, 0);
  ui->extendedTimelineContainer->getTimeline()->resetRange();
  ui->extendedTimelineContainer->getTimelineTicks()->clearProject();
}

void SeekBar::initialize() {
  if (!project || !project->isInit()) {
    return;
  }

  enableInternal();
  // Populate the list of curves.
  ui->curvesTree->populate(*project);
}

void SeekBar::cleanStitcher() {
  ui->extendedTimelineContainer->setEnabled(false);
  ui->extendedTimelineContainer->getTimeline()->setValue(0);
  ui->curvesTree->clear();
  ui->curvesTree->removePreviouslySelectedFromTimeline();

  setCurrentFrameLabel(0);
  firstBoundFrame = -1;
  lastBoundFrame = -1;
}

void SeekBar::enableInternal() {
  if (!project || !project->isInit()) {
    return;
  }
  ui->extendedTimelineContainer->getTimeline()->setBounds(project->getFirstFrame(), project->getLastFrame());
  ui->extendedTimelineContainer->enableInternal();
  refreshTimeWidgets();
}

void SeekBar::refreshTimeWidgets() {
  ui->firstFrameWidget->setFrame(project->getFirstFrame());
  ui->lastFrameWidget->setFrame(project->getLastFrame());
  updateSequence();
}

// called when a new frame has been displayed
void SeekBar::refresh(mtime_t date) {
  if (!project || !project->isInit()) {
    return;
  }

  VideoStitch::FrameRate frameRate = GlobalController::getInstance().getController()->getFrameRate();
  int curFrame = std::round((date / 1000000.0) * (frameRate.num / (double)frameRate.den));

  ui->extendedTimelineContainer->getTimeline()->setValue(curFrame);

  setCurrentFrameLabel(curFrame);

  if (curFrame >= getMaximumFrame()) {
    emit reqPause();
  }
}

void SeekBar::updateToPano(int lastStitchableFrame, int currentFrame) {
  VideoStitch::FrameRate frameRate = GlobalController::getInstance().getController()->getFrameRate();
  bool isLongerThanAnHour = TimeConverter::isLongerThanAnHour(lastStitchableFrame, frameRate);
  ui->firstFrameWidget->updateInputMask(isLongerThanAnHour, TimeConverter::hasMoreThanThreeIntDigits(frameRate));
  ui->lastFrameWidget->updateInputMask(isLongerThanAnHour, TimeConverter::hasMoreThanThreeIntDigits(frameRate));

  // set the timeline cursor to the project's initial value
  setCurrentFrameLabel(currentFrame);
  ui->extendedTimelineContainer->getTimeline()->setRange(0, lastStitchableFrame);
  ui->extendedTimelineContainer->getTimeline()->setValue(currentFrame);
  ui->firstFrameWidget->setMaxFrame(lastStitchableFrame);
  ui->lastFrameWidget->setMaxFrame(lastStitchableFrame);
}

void SeekBar::pause() { ui->playButton->setChecked(true); }

void SeekBar::play() { ui->playButton->setChecked(false); }

void SeekBar::seek(frameid_t frame) {
  frame = qBound(getMinimumFrame(), frame, getMaximumFrame());
  emit reqPause();
  emit reqSeek(comp->add(), frame);
}

void SeekBar::setWorkingArea(int firstFr, int lastFr) {
  project->setFirstFrame(firstFr);
  project->setLastFrame(lastFr);
  enableInternal();
}

// play/pause button (play is false, pause is true)
void SeekBar::on_playButton_clicked(bool checked) {
  if (checked) {
    emit reqPause();
  } else {
    emit reqPlay();
  }
}

void SeekBar::on_toStartButton_clicked() { startTimeEdited(getCurrentFrameFromCursorPosition()); }

void SeekBar::on_toStopButton_clicked() { stopTimeEdited(getCurrentFrameFromCursorPosition()); }

void SeekBar::startTimeEdited(int frame) {
  WorkingAreaChangedCommand *command = new WorkingAreaChangedCommand(project->getFirstFrame(), project->getLastFrame(),
                                                                     frame, project->getLastFrame(), this);
  qApp->findChild<QUndoStack *>()->push(command);
  updateSequence();
}

void SeekBar::stopTimeEdited(int frame) {
  WorkingAreaChangedCommand *command = new WorkingAreaChangedCommand(project->getFirstFrame(), project->getLastFrame(),
                                                                     project->getFirstFrame(), frame, this);
  qApp->findChild<QUndoStack *>()->push(command);
  updateSequence();
}

void SeekBar::leftShortcutCalled() {
  ui->extendedTimelineContainer->getTimeline()->moveCursorTo(ui->extendedTimelineContainer->getTimeline()->value() - 1);
}

void SeekBar::rightShortcutCalled() {
  ui->extendedTimelineContainer->getTimeline()->moveCursorTo(ui->extendedTimelineContainer->getTimeline()->value() + 1);
}

void SeekBar::nextKeyFrameShortcutCalled() {
  if (ui->extendedTimelineContainer->allowsKeyFrameNavigation()) {
    ui->extendedTimelineContainer->getTimeline()->moveCursorTo(
        ui->extendedTimelineContainer->getTimeline()->nextKeyFrame());
  }
}

void SeekBar::prevKeyFrameShortcutCalled() {
  if (ui->extendedTimelineContainer->allowsKeyFrameNavigation()) {
    ui->extendedTimelineContainer->getTimeline()->moveCursorTo(
        ui->extendedTimelineContainer->getTimeline()->prevKeyFrame());
  }
}
