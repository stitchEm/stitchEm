// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "timelinecontainer.hpp"
#include "ui_timelinecontainer.h"
#include "libvideostitch-gui/mainwindow/objectutil.hpp"
#include "timeline.hpp"
#include <QWheelEvent>

static const int NUM_OF_STEPS(15);

TimelineContainer::TimelineContainer(QWidget *parent) : QWidget(parent), ui(new Ui::TimelineContainer) {
  ui->setupUi(this);

  ui->timelineToolbar->setEnabled(false);
  ui->extendedTimeline->setEnabled(true);
  ui->playheadWidget->setEnabled(false);
  ui->playheadWidget->setTimeline(ui->extendedTimeline);

  connect(ui->extendedTimeline, SIGNAL(reqRefreshTicks()), ui->playheadWidget, SLOT(update()), Qt::DirectConnection);
  connect(ui->extendedTimeline, SIGNAL(reqUpdateZoomSliders(double)), this, SLOT(updateZoomSliders(double)));
  connect(ui->extendedTimeline, SIGNAL(reqUpdateKeyFrames(bool)), this, SLOT(enableKeyFrameNavigation(bool)));
  connectSliders();
  ui->extendedTimeline->viewport()->installEventFilter(this);
  enableKeyFrameNavigation(false);
}

void TimelineContainer::enableInternal() {
  ui->timelineToolbar->setEnabled(true);
  ui->extendedTimeline->setEnabled(true);
  ui->playheadWidget->setEnabled(true);
}

void TimelineContainer::enableKeyFrameNavigation(bool enable) {
  ui->nextKeyFrameButton->setEnabled(enable);
  ui->prevKeyFrameButton->setEnabled(enable);
}

Timeline *TimelineContainer::getTimeline() const { return ui->extendedTimeline; }

TimelineTicks *TimelineContainer::getTimelineTicks() const { return ui->playheadWidget; }

TimelineContainer::~TimelineContainer() { delete ui; }

void TimelineContainer::updateZoomSliders(double maxZoomX) {
  disconnectSliders();
  QMatrix matrix = ui->extendedTimeline->matrix();
  int currentNameOfSteps =
      (int)log(maxZoomX) - (int)log(matrix.m11());  // don't simplify it since we need to round each term

  const double mult = (NUM_OF_STEPS) / (double)currentNameOfSteps;

  ui->extendedTimeline->setZoomMultiplier(mult);
  ui->xZoomSlider->setMinimum(log(matrix.m11()) * mult);
  ui->xZoomSlider->setMaximum(log(maxZoomX) * mult);
  ui->xZoomSlider->setValue(ui->xZoomSlider->minimum());
  connectSliders();
}

void TimelineContainer::connectSliders(bool state) {
  VideoStitch::Helper::toggleConnect(state, ui->xZoomSlider, SIGNAL(valueChanged(int)), ui->extendedTimeline,
                                     SLOT(setZoomLevel(int)), Qt::UniqueConnection);
}

void TimelineContainer::disconnectSliders() { connectSliders(false); }

void TimelineContainer::on_plusButton_clicked() {
  ui->xZoomSlider->setValue(ui->xZoomSlider->value() + ui->xZoomSlider->singleStep());
}

void TimelineContainer::on_minusButton_clicked() {
  ui->xZoomSlider->setValue(ui->xZoomSlider->value() - ui->xZoomSlider->singleStep());
}

void TimelineContainer::on_prevKeyFrameButton_clicked() {
  ui->extendedTimeline->moveCursorTo(ui->extendedTimeline->prevKeyFrame());
}

void TimelineContainer::on_nextKeyFrameButton_clicked() {
  ui->extendedTimeline->moveCursorTo(ui->extendedTimeline->nextKeyFrame());
}

bool TimelineContainer::eventFilter(QObject *watched, QEvent *event) {
  bool ret = false;
  if (watched == ui->extendedTimeline->viewport()) {
    if (event->type() == QEvent::Wheel && dynamic_cast<QWheelEvent *>(event)->modifiers() == Qt::ControlModifier) {
      QApplication::sendEvent(ui->xZoomSlider, event);
      ret = true;
    }
  }
  return ret;
}

QSize TimelineContainer::sizeHint() const { return QSize(0, 100); }

bool TimelineContainer::allowsKeyFrameNavigation() const {
  return ui->nextKeyFrameButton->isEnabled() && ui->prevKeyFrameButton->isEnabled();
}
