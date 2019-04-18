// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "croprectangleeditor.hpp"
#include <QMouseEvent>
#include <QPainter>

CropRectangleEditor::CropRectangleEditor(const QSize thumbnailSize, const QSize frameSize, const Crop &initCrop,
                                         QWidget *const parent)
    : CropShapeEditor(thumbnailSize, frameSize, initCrop, parent) {}

void CropRectangleEditor::setDefaultCrop() {
  Q_ASSERT(getRatio() > 0);
  shape.setTop(0);
  shape.setLeft(0);
  shape.setHeight(thumbnailSize.height());
  shape.setWidth(thumbnailSize.width());
  update();
}

const QRectF CropRectangleEditor::getTopArea(const QRectF rectangle) const {
  return QRectF(rectangle.left() + SEL_OFFSET, rectangle.top() - SEL_OFFSET, rectangle.width() - PEN_THICK * SEL_OFFSET,
                2 * SEL_OFFSET);
}

const QRectF CropRectangleEditor::getBottomArea(const QRectF rectangle) const {
  return QRectF(rectangle.left() + SEL_OFFSET, rectangle.bottom() - SEL_OFFSET,
                rectangle.width() - PEN_THICK * SEL_OFFSET, 2 * SEL_OFFSET);
}

const QRectF CropRectangleEditor::getLeftArea(const QRectF rectangle) const {
  return QRectF(rectangle.left() - SEL_OFFSET, rectangle.top() + SEL_OFFSET, 2 * SEL_OFFSET,
                rectangle.height() - PEN_THICK * SEL_OFFSET);
}

const QRectF CropRectangleEditor::getRightArea(const QRectF rectangle) const {
  return QRectF(rectangle.right() - SEL_OFFSET, rectangle.top() + SEL_OFFSET, 2 * SEL_OFFSET,
                rectangle.height() - PEN_THICK * SEL_OFFSET);
}

const QRectF CropRectangleEditor::getTopLeftCorner(const QRectF rectangle) const {
  return QRectF(rectangle.left() - SEL_OFFSET, rectangle.top() - SEL_OFFSET, 2 * SEL_OFFSET, 2 * SEL_OFFSET);
}

const QRectF CropRectangleEditor::getTopRightCorner(const QRectF rectangle) const {
  return QRectF(rectangle.right() - SEL_OFFSET, rectangle.top() - SEL_OFFSET, 2 * SEL_OFFSET, 2 * SEL_OFFSET);
}

const QRectF CropRectangleEditor::getBottomLeftCorner(const QRectF rectangle) const {
  return QRectF(rectangle.left() - SEL_OFFSET, rectangle.bottom() - SEL_OFFSET, 2 * SEL_OFFSET, 2 * SEL_OFFSET);
}

const QRectF CropRectangleEditor::getBottomRightCorner(const QRectF rectangle) const {
  return QRectF(rectangle.right() - SEL_OFFSET, rectangle.bottom() - SEL_OFFSET, 2 * SEL_OFFSET, 2 * SEL_OFFSET);
}

void CropRectangleEditor::drawCropShape(QPainter &painter) {
  QPainterPath background;
  background.setFillRule(Qt::WindingFill);
  background.addRect(rect());

  QPainterPath rectangle;
  rectangle.addRect(shape);
  const QPainterPath intersection = background.subtracted(rectangle);
  painter.setOpacity(opacity);
  painter.fillPath(intersection, currentFill);
  painter.setOpacity(1.0);
  painter.strokePath(rectangle.simplified(), border);
}

void CropRectangleEditor::mouseMoveEvent(QMouseEvent *event) {
  if (ignoreEvent) {
    event->ignore();
    return;
  }

  switch (modificationMode) {
    case ModificationMode::NoModification:
      break;
    case ModificationMode::ResizeFromTop:
      shape.setTop(qMin(qreal(event->pos().y()), shape.bottom()));
      break;
    case ModificationMode::ResizeFromBottom:
      shape.setBottom(qMax(qreal(event->pos().y()), shape.top()));
      break;
    case ModificationMode::ResizeFromLeft:
      shape.setLeft(qMin(qreal(event->pos().x()), shape.right()));
      break;
    case ModificationMode::ResizeFromRight:
      shape.setRight(qMax(qreal(event->pos().x()), shape.left()));
      break;
    case ModificationMode::ResizeFromTopLeft:
      shape.setTop(qMin(qreal(event->pos().y()), shape.bottom()));
      shape.setLeft(qMin(qreal(event->pos().x()), shape.right()));
      break;
    case ModificationMode::ResizeFromTopRight:
      shape.setTop(qMin(qreal(event->pos().y()), shape.bottom()));
      shape.setRight(qMax(qreal(event->pos().x()), shape.left()));
      break;
    case ModificationMode::ResizeFromBottomLeft:
      shape.setBottom(qMax(qreal(event->pos().y()), shape.top()));
      shape.setLeft(qMin(qreal(event->pos().x()), shape.right()));
      break;
    case ModificationMode::ResizeFromBottomRight:
      shape.setBottom(qMax(qreal(event->pos().y()), shape.top()));
      shape.setRight(qMax(qreal(event->pos().x()), shape.left()));
      break;
    case ModificationMode::Move:
      shape.moveCenter(event->pos() - distanceToCenter);
      break;
  }

  if (modificationMode != ModificationMode::NoModification) {
    update();
    emit notifyCropSet(getCrop());
  }

  if (getTopArea(shape).contains(event->pos()) || getBottomArea(shape).contains(event->pos())) {
    setCursor(QCursor(Qt::CursorShape::SizeVerCursor));
  } else if (getLeftArea(shape).contains(event->pos()) || getRightArea(shape).contains(event->pos())) {
    setCursor(QCursor(Qt::CursorShape::SizeHorCursor));
  } else if (getTopLeftCorner(shape).contains(event->pos()) || getBottomRightCorner(shape).contains(event->pos())) {
    setCursor(QCursor(Qt::CursorShape::SizeFDiagCursor));
  } else if (getTopRightCorner(shape).contains(event->pos()) || getBottomLeftCorner(shape).contains(event->pos())) {
    setCursor(QCursor(Qt::CursorShape::SizeBDiagCursor));
  } else if (getCentralArea(shape).contains(event->pos())) {
    setCursor(QCursor(Qt::CursorShape::SizeAllCursor));
  } else {
    setCursor(QCursor(Qt::CursorShape::ArrowCursor));
  }
}

void CropRectangleEditor::mousePressEvent(QMouseEvent *event) {
  if (ignoreEvent) {
    event->ignore();
    return;
  }
  if (getTopArea(shape).contains(event->pos())) {
    modificationMode = ModificationMode::ResizeFromTop;
  } else if (getBottomArea(shape).contains(event->pos())) {
    modificationMode = ModificationMode::ResizeFromBottom;
  } else if (getLeftArea(shape).contains(event->pos())) {
    modificationMode = ModificationMode::ResizeFromLeft;
  } else if (getRightArea(shape).contains(event->pos())) {
    modificationMode = ModificationMode::ResizeFromRight;
  } else if (getTopLeftCorner(shape).contains(event->pos())) {
    modificationMode = ModificationMode::ResizeFromTopLeft;
  } else if (getTopRightCorner(shape).contains(event->pos())) {
    modificationMode = ModificationMode::ResizeFromTopRight;
  } else if (getBottomLeftCorner(shape).contains(event->pos())) {
    modificationMode = ModificationMode::ResizeFromBottomLeft;
  } else if (getBottomRightCorner(shape).contains(event->pos())) {
    modificationMode = ModificationMode::ResizeFromBottomRight;
  } else if (getCentralArea(shape).contains(event->pos())) {
    modificationMode = ModificationMode::Move;
    distanceToCenter = event->pos() - shape.center();
  }
}

void CropRectangleEditor::mouseReleaseEvent(QMouseEvent *event) {
  if (ignoreEvent) {
    event->ignore();
    return;
  }
  modificationMode = ModificationMode::NoModification;
}
