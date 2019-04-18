// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "timelineticks.hpp"
#include "timeline.hpp"

#include "videostitcher/globalpostprodcontroller.hpp"

#include "libvideostitch-gui/mainwindow/timeconverter.hpp"
#include "libvideostitch-gui/widgets/vsgraphics.hpp"

#include <QApplication>
#include <QMouseEvent>

static const int GRAB_PRECISION(5);
static const QColor COLOR_TICK(255, 166, 12);
static const QColor COLOR_HANDLE(150, 150, 150);
static const QColor COLOR_HANDLE_BORDER(33, 33, 33);

static const VideoStitch::FrameRate getDefaultFrameRate() { return {100, 1}; }

TimelineTicks::TimelineTicks(QWidget *parent)
    : IProcessWidget(parent),
      tickColor(COLOR_TICK),
      boundHandleColor(COLOR_HANDLE),
      handleSize(10),
      boundHandleSize(10),
      tickLength(10),
      subTickLength(tickLength / 2),
      frameLabelMargin(100),
      timeline(nullptr),
      movingType(MouseMovingType::NoMoving) {
  setMouseTracking(true);
}

void TimelineTicks::setTimeline(Timeline *t) { this->timeline = t; }

void TimelineTicks::setTickColor(QColor color) { tickColor = color; }

QColor TimelineTicks::getTickColor() const { return tickColor; }

void TimelineTicks::setBoundHandleColor(QColor color) { boundHandleColor = color; }

QColor TimelineTicks::getBoundHandleColor() const { return boundHandleColor; }

void TimelineTicks::setHandleSize(int size) { handleSize = size; }

int TimelineTicks::getHandleSize() const { return handleSize; }

void TimelineTicks::setBoundHandleSize(int size) { boundHandleSize = size; }

int TimelineTicks::getBoundHandleSize() const { return boundHandleSize; }

void TimelineTicks::setTickLength(int length) { tickLength = length; }

int TimelineTicks::getTickLength() const { return tickLength; }

void TimelineTicks::setSubTickLength(int length) { subTickLength = length; }

int TimelineTicks::getSubTickLength() const { return subTickLength; }

void TimelineTicks::setFrameLabelMargin(int margin) { frameLabelMargin = margin; }

int TimelineTicks::getFrameLabelMargin() const { return frameLabelMargin; }

void TimelineTicks::mousePressEvent(QMouseEvent *event) {
  if (event->button() != Qt::LeftButton) {
    return;
  }
  if (event->x() > timeline->viewport()->width()) {
    return;
  }
  setCursor(Qt::ClosedHandCursor);

  const int curFramePos = timeline->getCursorPosition();
  const int lowerBound = timeline->getLowerBoundPosition();
  const int upperBound = timeline->getUpperBoundPosition();

  if (isMouseCloseToHandle(event->x(), curFramePos) ||
      (!isMouseCloseToHandle(event->x(), lowerBound) && !isMouseCloseToHandle(event->x(), upperBound))) {
    movingType = MouseMovingType::MainCursorMoving;
    moveCursorFromPosition(event->x());
  } else if (isMouseCloseToHandle(event->x(), upperBound)) {
    movingType = MouseMovingType::UpperBoundMoving;
  } else {
    movingType = MouseMovingType::LowerBoundMoving;
  }
}

void TimelineTicks::mouseMoveEvent(QMouseEvent *event) {
  if (event->x() < 0 || event->x() > width()) {
    return;
  }
  if (event->x() > timeline->viewport()->width()) {
    return;
  }

  switch (movingType) {
    case MouseMovingType::MainCursorMoving: {
      moveCursorFromPosition(event->x());
      break;
    }
    case MouseMovingType::LowerBoundMoving: {
      const int position = qMax(0, qMin(event->x(), width()));
      const frameid_t frame =
          qBound(frameid_t(timeline->minimum()), mapFromPosition(position), frameid_t(timeline->maximum()));
      emit lowerBoundHandlePositionChanged(frame);
      break;
    }
    case MouseMovingType::UpperBoundMoving: {
      const int position = qMax(0, qMin(event->x(), width()));
      const frameid_t frame =
          qBound(frameid_t(timeline->minimum()), mapFromPosition(position), frameid_t(timeline->maximum()));
      emit upperBoundHandlePositionChanged(frame);
      break;
    }
    case MouseMovingType::NoMoving: {
      const int curFramePos = timeline->getCursorPosition();
      const int lowerBound = timeline->getLowerBoundPosition();
      const int upperBound = timeline->getUpperBoundPosition();
      if (isMouseCloseToHandle(event->x(), curFramePos) || isMouseCloseToHandle(event->x(), lowerBound) ||
          isMouseCloseToHandle(event->x(), upperBound)) {
        setCursor(Qt::OpenHandCursor);
      } else {
        setCursor(Qt::PointingHandCursor);
      }
      break;
    }
  }
}

void TimelineTicks::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() != Qt::LeftButton) {
    return;
  }
  movingType = MouseMovingType::NoMoving;
  setCursor(Qt::OpenHandCursor);
}

void TimelineTicks::mouseDoubleClickEvent(QMouseEvent *event) {
  if (!isMouseCloseToHandle(event->x(), timeline->getCursorPosition())) {
    if (isMouseCloseToHandle(event->x(), timeline->getLowerBoundPosition())) {
      emit lowerBoundHandlePositionChanged(timeline->minimum());
    } else if (isMouseCloseToHandle(event->x(), timeline->getUpperBoundPosition())) {
      emit upperBoundHandlePositionChanged(timeline->maximum());
    }
  }
}

void TimelineTicks::paintHandle(QPainter &painter, int position, int origin, int size, const QColor color) {
  QPainterPath path;
  path.moveTo(position, origin);
  path.lineTo(position - size / 2.0f, -size / 2.0f + origin);
  path.lineTo(position - size / 2.0f, -size + origin);
  path.lineTo(position + size / 2.0f, -size + origin);
  path.lineTo(position + size / 2.0f, -size / 2.0f + origin);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.setPen(QPen(COLOR_HANDLE_BORDER, 0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  painter.setBrush(QBrush(color));
  painter.drawPath(path);
  painter.setRenderHint(QPainter::Antialiasing, false);
}

bool TimelineTicks::isMouseCloseToHandle(int mousePosition, int handlePosition) const {
  return mousePosition - GRAB_PRECISION <= handlePosition && mousePosition + GRAB_PRECISION >= handlePosition;
}

frameid_t TimelineTicks::mapFromPosition(int position) const {
  const int firstVisibleFrame = timeline->mapFromPosition(timeline->visibleRect().left());
  const int lastVisibleFrame = timeline->mapFromPosition(timeline->visibleRect().right());
  const int frame = firstVisibleFrame + std::round(double(position * (lastVisibleFrame - firstVisibleFrame)) /
                                                   double(timeline->viewport()->width()));
  return frameid_t(frame);
}

void TimelineTicks::paintEvent(QPaintEvent *event) {
  Q_UNUSED(event)
  Q_ASSERT(timeline != nullptr);

  QPainter painter(this);
  drawBackground(painter);

  drawTicks(painter);

  // Paint the lower bound handle
  drawSequenceHandler(timeline->getLowerBoundPosition(), painter);
  // Paint the upper bound handle
  drawSequenceHandler(timeline->getUpperBoundPosition(), painter);
  // Paint the player cursor line and its handle
  drawPlayerHandler(painter);
}

void TimelineTicks::drawTicks(QPainter &painter) {
  painter.setPen(tickColor);

  const VideoStitch::FrameRate frameRate = getFrameRateToUse();
  if (frameRate.num <= 0 || frameRate.den <= 0) {
    return;  // if we're here, something may not be set.
  }

  int mainStep = 1;  // Number of frames between 2 ticks
  int subStep = 1;   // Number of frames between 2 sub-ticks
  computeSteps(mainStep, subStep);

  // Compute the first and last labels to paint. Even if it is not included in the timeline, the text can be visible
  const int firstVisibleFrame = timeline->mapFromPosition(timeline->visibleRect().left());
  const int lastVisibleFrame = timeline->mapFromPosition(timeline->visibleRect().right());
  const frameid_t firstLabeledFrame = firstVisibleFrame - firstVisibleFrame % mainStep;
  const frameid_t lastLabeledFrame = lastVisibleFrame + mainStep - lastVisibleFrame % mainStep;
  // We can have skipped frames if the framerate is not an interger, but we don't care if we display all the frames
  bool manageSkippedFrames = (frameRate.num % frameRate.den != 0) && mainStep > 1;

  // Paint the ticks and sub-ticks
  for (int subStepFrame = firstLabeledFrame; subStepFrame <= lastLabeledFrame; subStepFrame += subStep) {
    // If not in the range, we don't want to paint the label
    if (subStepFrame > timeline->maximum() || subStepFrame < timeline->minimum()) {
      continue;
    }

    // In case of a main step, we will paint the line and the label
    if (subStepFrame % mainStep == 0) {
      // Compute possible skipped frames and paint the line
      int skippedFrames = manageSkippedFrames ? subStepFrame / frameRate.den : 0;
      frameid_t realFrame = subStepFrame - skippedFrames;
      drawTickLineAndLabel(painter, realFrame, frameRate);
    }

    // In case of a sub-tick, we only paint a line
    if (subStep != mainStep) {
      // When the framerate is not an integer (47.95, 29.97, ...), this case is not exclusive with the previous one
      // So we still want to draw a sub-tick
      drawSubTickLine(painter, subStepFrame);
    }
  }
}

void TimelineTicks::drawTickLineAndLabel(QPainter &painter, const frameid_t frame, VideoStitch::FrameRate frameRate) {
  const int viewportPosition = timeline->mapFromScene(timeline->mapToPosition(frame), 0).x();
  painter.drawLine(viewportPosition, 0, viewportPosition, tickLength);

  QString frameLabel = TimeConverter::frameToTimeDisplay(frame, frameRate);
  QString frameNumber = frameLabel.right(frameLabel.size() - frameLabel.lastIndexOf(":") - 1);
  if (frameNumber.toInt() != 0) {
    frameLabel = frameNumber + "f";
  }
  const int labelWidth = QApplication::fontMetrics().width(frameLabel);
  painter.drawText(QPointF(viewportPosition - labelWidth / 2, tickLength + frameLabelMargin), frameLabel);
}

void TimelineTicks::drawSubTickLine(QPainter &painter, const frameid_t frame) {
  const int viewportPosition = timeline->mapFromScene(timeline->mapToPosition(frame), 0).x();
  painter.drawLine(viewportPosition, 0, viewportPosition, subTickLength);
}

void TimelineTicks::drawBackground(QPainter &painter) {
  painter.setPen(Qt::NoPen);
  painter.setBrush(VSGraphicsScene::backgroundColor);
  painter.drawRect(0, 0, width(), height());
}

void TimelineTicks::computeSteps(int &mainStep, int &subStep) const {
  const VideoStitch::FrameRate frameRate = getFrameRateToUse();
  const int upperFramerate = std::ceil(double(frameRate.num) / double(frameRate.den));
  const QVector<int> frameScales = computeFrameScales(upperFramerate);

  mainStep = frameScales.first();
  subStep = frameScales.first();

  const int firstVisibleFrame = timeline->mapFromPosition(timeline->visibleRect().left());
  const int lastVisibleFrame = timeline->mapFromPosition(timeline->visibleRect().right());
  const int displayedFrames = lastVisibleFrame - firstVisibleFrame;

  for (int scale : frameScales) {
    if (scale <= displayedFrames) {
      subStep = mainStep;
      mainStep = scale;
    } else {
      break;
    }
  }
}

VideoStitch::FrameRate TimelineTicks::getFrameRateToUse() const {
  if (project && GlobalController::getInstance().getController() != nullptr) {
    return GlobalController::getInstance().getController()->getFrameRate();
  } else {
    return getDefaultFrameRate();
  }
}

QVector<int> TimelineTicks::computeFrameScales(int upperFramerate) {
  QVector<int> frameScales;  // unit = frames
  // 1 frame
  frameScales << 1;

  // This scale depends of the framerate, because every scale should be a multiple of the previous scale
  static const QMap<int, int> intermediateScale = {{15, 5},  {24, 4},  {25, 5},   {30, 5},  {48, 8},
                                                   {50, 10}, {60, 10}, {100, 10}, {120, 10}};
  if (intermediateScale.contains(upperFramerate)) {
    frameScales << intermediateScale.value(upperFramerate);
  }

  // 1 second, 10 seconds, 1 minute, 10 minutes, 1 hour
  frameScales << upperFramerate << 10 * upperFramerate << 60 * upperFramerate << 600 * upperFramerate
              << 3600 * upperFramerate;
  return frameScales;
}

void TimelineTicks::moveCursorFromPosition(int position) { timeline->moveCursorTo(mapFromPosition(position)); }

void TimelineTicks::drawSequenceHandler(const int position, QPainter &painter) {
  paintHandle(painter, position, height(), boundHandleSize, boundHandleColor);
}

void TimelineTicks::drawPlayerHandler(QPainter &painter) {
  const int curFramePos = timeline->getCursorPosition();
  painter.setPen(QPen(timeline->getCursorLineColor(), 0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  painter.drawLine(curFramePos, 0, curFramePos, height());
  paintHandle(painter, curFramePos, handleSize, handleSize, timeline->getCursorLineColor());
}
