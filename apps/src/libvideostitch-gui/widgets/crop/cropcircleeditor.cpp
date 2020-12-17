// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "cropcircleeditor.hpp"
#include "libvideostitch/imageProcessingUtils.hpp"
#include "libvideostitch-gui/mainwindow/vssettings.hpp"
#include <math.h>
#include <QPainter>
#include <QPainterPath>
#include <QMouseEvent>

#define PRECISION 0.1f

CropCircleEditor::CropCircleEditor(const QSize thumbnailSize, const QSize frameSize, const Crop& initCrop,
                                   QWidget* const parent)
    : CropShapeEditor(thumbnailSize, frameSize, initCrop, parent), haveToResizeCircle(false) {
  setDefaultCrop();
  connect(&asyncAutoCropTaskWatcher, SIGNAL(finished()), this, SLOT(update()));
}

void CropCircleEditor::setDefaultCircle() {
  shape.setTop(0);
  shape.setLeft(thumbnailSize.width() / 2 - thumbnailSize.height() / 2);
  shape.setHeight(thumbnailSize.height());
  shape.setWidth(thumbnailSize.height());
}

void CropCircleEditor::findAutoCrop() {
  int x = 0;
  int y = 0;
  int radius = 0;
  if (VideoStitch::Util::ImageProcessing::findCropCircle(cachedScaledFrame.width(), cachedScaledFrame.height(),
                                                         cachedScaledFrame.bits(), x, y, radius)
          .ok()) {
    shape.setTop(y - radius);
    shape.setLeft(x - radius);
    shape.setHeight(radius * 2);
    shape.setWidth(radius * 2);
  } else {
    setDefaultCircle();
  }
}

void CropCircleEditor::setDefaultCrop() {
  Q_ASSERT(getRatio() > 0);
  // Perform detection only when data was passed
  if (VSSettings::getSettings() && VSSettings::getSettings()->getShowExperimentalFeatures()) {
    if (!asyncAutoCropTask.isRunning()) {
      asyncAutoCropTask = QtConcurrent::run(std::bind(&CropCircleEditor::findAutoCrop, this));
      asyncAutoCropTaskWatcher.setFuture(asyncAutoCropTask);
    }
  } else {
    setDefaultCircle();
  }
  update();
}

bool CropCircleEditor::isValidCrop() const {
  if (shape.left() == 0 && shape.top() == 0 && shape.width() == thumbnailSize.width() &&
      shape.height() == thumbnailSize.height()) {
    return false;
  }
  return true;
}

bool CropCircleEditor::isAutoCropSupported() const { return true; }

bool CropCircleEditor::pointInBorder(const QRectF& rectangle, const QPoint& point) const {
  const float squareRadius = rectangle.width() * rectangle.width() / 4.0f;
  const float squareDistance = QVector2D(point - rectangle.center()).lengthSquared();
  return (fabs(squareDistance / squareRadius - 1.) < PRECISION);
}

bool CropCircleEditor::pointInCircle(const QRectF& rectangle, const QPoint& point) const {
  const float squareRadius = rectangle.width() * rectangle.width() / 4.0f;
  const float squareDistance = QVector2D(point - rectangle.center()).lengthSquared();
  return (squareDistance < squareRadius);
}

bool CropCircleEditor::pointInBottomLeftArea(const QRectF& rectangle, const QPoint& point) const {
  return point.x() < rectangle.center().x() && point.y() >= rectangle.center().y();
}
bool CropCircleEditor::pointInBottomRightArea(const QRectF& rectangle, const QPoint& point) const {
  return point.x() >= rectangle.center().x() && point.y() >= rectangle.center().y();
}
bool CropCircleEditor::pointInTopLeftArea(const QRectF& rectangle, const QPoint& point) const {
  return point.x() < rectangle.center().x() && point.y() < rectangle.center().y();
}
bool CropCircleEditor::pointInTopRightArea(const QRectF& rectangle, const QPoint& point) const {
  return point.x() >= rectangle.center().x() && point.y() < rectangle.center().y();
}

void CropCircleEditor::mousePressEvent(QMouseEvent* event) {
  if (ignoreEvent) {
    event->ignore();
    return;
  }
  if (pointInBorder(shape, event->pos())) {
    haveToResizeCircle = true;
    return;
  }
  if (pointInCircle(shape, event->pos())) {
    modificationMode = ModificationMode::Move;
    distanceToCenter = event->pos() - shape.center();
  }
}

void CropCircleEditor::mouseReleaseEvent(QMouseEvent* event) {
  if (ignoreEvent) {
    event->ignore();
    return;
  }
  haveToResizeCircle = false;
  modificationMode = ModificationMode::NoModification;
}

void CropCircleEditor::setCursorWhenMoving(const QPoint& cursorPos) {
  const float distanceX = fabs((float)cursorPos.x() - shape.center().x());
  const float distanceY = fabs((float)shape.center().y() - cursorPos.y());
  const int circleRadius = QVector2D(cursorPos - shape.center()).length();
  const float coeff = 0.5f;  // sinus(30°) == cosinus(60°) == 0.5
  if (pointInBorder(shape, cursorPos)) {
    if (distanceX < circleRadius * coeff) {
      setCursor(QCursor(Qt::CursorShape::SizeVerCursor));
    } else if (distanceY < circleRadius * coeff) {
      setCursor(QCursor(Qt::CursorShape::SizeHorCursor));
    } else if (pointInTopRightArea(shape, cursorPos) || pointInBottomLeftArea(shape, cursorPos)) {
      setCursor(QCursor(Qt::CursorShape::SizeBDiagCursor));
    } else if (pointInTopLeftArea(shape, cursorPos) || pointInBottomRightArea(shape, cursorPos)) {
      setCursor(QCursor(Qt::CursorShape::SizeFDiagCursor));
    }
  } else if (pointInCircle(shape, cursorPos)) {
    setCursor(QCursor(Qt::CursorShape::SizeAllCursor));
  } else {
    setCursor(QCursor(Qt::CursorShape::ArrowCursor));
  }
}

void CropCircleEditor::mouseMoveEvent(QMouseEvent* event) {
  if (ignoreEvent) {
    event->ignore();
    return;
  }
  if (haveToResizeCircle) {
    if (pointInTopLeftArea(shape, event->pos())) {
      modificationMode = ModificationMode::ResizeFromTopLeft;
    } else if (pointInTopRightArea(shape, event->pos())) {
      modificationMode = ModificationMode::ResizeFromTopRight;
    } else if (pointInBottomLeftArea(shape, event->pos())) {
      modificationMode = ModificationMode::ResizeFromBottomLeft;
    } else {
      modificationMode = ModificationMode::ResizeFromBottomRight;
    }
  }
  const int centerX = shape.center().x();
  const int centerY = shape.center().y();
  const int circleRadius = QVector2D(event->pos() - shape.center()).length();
  switch (modificationMode) {
    case ModificationMode::ResizeFromTopRight:
      shape.setTop(qMin(qreal(centerY - circleRadius), shape.bottom()));
      shape.setWidth(shape.height());
      break;
    case ModificationMode::ResizeFromTopLeft:
      shape.setTop(qMin(qreal(centerY - circleRadius), shape.bottom()));
      shape.setLeft(qMin(qreal(centerX - circleRadius), shape.right()));
      break;
    case ModificationMode::ResizeFromBottomLeft:
      shape.setBottom(qMax(qreal(centerY + circleRadius), shape.top()));
      shape.setLeft(qMin(qreal(centerX - circleRadius), shape.right()));
      break;
    case ModificationMode::ResizeFromBottomRight:
      shape.setBottom(qMax(qreal(centerY + circleRadius), shape.top()));
      shape.setWidth(shape.height());
      break;
    case ModificationMode::Move:
      shape.moveCenter(event->pos() - distanceToCenter);
      break;
    case ModificationMode::NoModification:
    default:
      break;
  }

  if (modificationMode != ModificationMode::NoModification) {
    update();
    emit notifyCropSet(getCrop());
  }
  setCursorWhenMoving(event->pos());
}

const QRectF CropCircleEditor::getTopArea(const QRectF rectangle) const {
  return QRectF(rectangle.center().x() - SEL_OFFSET, rectangle.y() - SEL_OFFSET, 2 * SEL_OFFSET, 2 * SEL_OFFSET);
}

const QRectF CropCircleEditor::getBottomArea(const QRectF rectangle) const {
  return QRectF(rectangle.center().x() - SEL_OFFSET, rectangle.bottom() - SEL_OFFSET, 2 * SEL_OFFSET, 2 * SEL_OFFSET);
}

const QRectF CropCircleEditor::getLeftArea(const QRectF rectangle) const {
  return QRectF(rectangle.center().x() - rectangle.width() / 2 - SEL_OFFSET, rectangle.center().y() - SEL_OFFSET,
                2 * SEL_OFFSET, 2 * SEL_OFFSET);
}

const QRectF CropCircleEditor::getRightArea(const QRectF rectangle) const {
  return QRectF(rectangle.center().x() + rectangle.width() / 2 - SEL_OFFSET, rectangle.center().y() + SEL_OFFSET,
                2 * SEL_OFFSET, 2 * SEL_OFFSET);
}

const QRectF CropCircleEditor::getTopLeftCorner(const QRectF) const { return QRectF(); }

const QRectF CropCircleEditor::getTopRightCorner(const QRectF) const { return QRectF(); }

const QRectF CropCircleEditor::getBottomLeftCorner(const QRectF) const { return QRectF(); }

const QRectF CropCircleEditor::getBottomRightCorner(const QRectF) const { return QRectF(); }

void CropCircleEditor::drawCropShape(QPainter& painter) {
  QPainterPath background;
  background.setFillRule(Qt::WindingFill);
  background.addRect(rect());

  QPainterPath circle;
  circle.addEllipse(shape);
  const QPainterPath intersection = background.subtracted(circle);
  painter.setOpacity(opacity);
  painter.fillPath(intersection, currentFill);
  painter.setOpacity(1.0);
  painter.strokePath(circle.simplified(), border);
}
