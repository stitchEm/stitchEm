// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "cropshapeeditor.hpp"
#include <QPainter>
#include <QMouseEvent>
#include <QtCore/qmath.h>
#include <functional>

static const unsigned int CROSS_LARGE(4);
static const float BACK_OPACITY(0.3f);
static const int MIN_SIDE(50);
static const int WHEEL_STEP(3);

CropShapeEditor::CropShapeEditor(const QSize thumbSize, const QSize frmSize, const Crop &initCrop, QWidget *parent)
    : QWidget(parent),
      frameSize(frmSize),
      thumbnailSize(thumbSize),
      shape(cropToShape(initCrop)),
      currentLine(lineColor),
      currentFill(fillColor),
      border(QPen(lineColor, PEN_THICK, Qt::DashLine, Qt::FlatCap, Qt::MiterJoin)),
      opacity(BACK_OPACITY),
      modificationMode(ModificationMode::NoModification),
      ignoreEvent(false) {
  setAttribute(Qt::WA_Hover);
  setBackgroundRole(QPalette::Base);
  setMouseTracking(true);
  setFixedWidth(thumbSize.width());
  setFixedHeight(thumbSize.height());
  setAttribute(Qt::WA_TranslucentBackground);
}

void CropShapeEditor::onResetToDefault() {
  setDefaultCrop();
  emit notifyCropSet(getCrop());
}

const QRectF CropShapeEditor::cropToShape(const Crop &crop) const {
  const float ratio = getRatio();
  Q_ASSERT(ratio > 0);
  return QRectF(crop.crop_left / ratio, crop.crop_top / ratio, (crop.crop_right - crop.crop_left) / ratio,
                (crop.crop_bottom - crop.crop_top) / ratio);
}

CropShapeEditor::~CropShapeEditor() {}

bool CropShapeEditor::isValidCrop() const { return true; }

void CropShapeEditor::disableEdition(const bool disable) {
  ignoreEvent = disable;
  currentFill = disable ? disableColor : fillColor;
  currentLine = disable ? disableColor : lineColor;
  border.setColor(currentLine);
}

void CropShapeEditor::setLineColor(QColor color) { lineColor = color; }

QColor CropShapeEditor::getLineColor() const { return lineColor; }

void CropShapeEditor::setFillColor(QColor color) { fillColor = color; }

QColor CropShapeEditor::getFillColor() const { return fillColor; }

void CropShapeEditor::setDisableColor(QColor color) { disableColor = color; }

QColor CropShapeEditor::getDisableColor() const { return disableColor; }

void CropShapeEditor::wheelEvent(QWheelEvent *event) {
  if (ignoreEvent) {
    event->ignore();
    return;
  }
  if (getCentralArea(shape).contains(event->pos())) {
    const int delta = qAbs(event->delta()) / WHEEL_STEP;
    const int adjust = 2 * delta;
    if (event->delta() > 0) {
      shape.adjust(-delta, -delta, delta, delta);
      update();
      emit notifyCropSet(getCrop());
    } else {
      if (shape.height() - adjust > MIN_SIDE && shape.width() - adjust > MIN_SIDE) {
        shape.adjust(delta, delta, -delta, -delta);
        update();
        emit notifyCropSet(getCrop());
      }
    }
  }
}

void CropShapeEditor::showEvent(QShowEvent *event) {
  currentFill = ignoreEvent ? disableColor : fillColor;
  currentLine = ignoreEvent ? disableColor : lineColor;
  border.setColor(currentLine);
  QWidget::showEvent(event);
}

void CropShapeEditor::drawCenterCross(QPainter &painter) {
  painter.drawLine(shape.center() - QPointF(0, CROSS_LARGE), shape.center() + QPointF(0, CROSS_LARGE));
  painter.drawLine(shape.center() - QPointF(CROSS_LARGE, 0), shape.center() + QPointF(CROSS_LARGE, 0));
}

void CropShapeEditor::paintEvent(QPaintEvent *event) {
  Q_UNUSED(event);
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.setPen(QPen(currentLine));
  drawCenterCross(painter);
  drawCropShape(painter);
}

const Crop CropShapeEditor::getCrop() const {
  const float ratio = getRatio();
  return Crop(qRound(shape.left() * ratio), qRound((shape.x() + shape.width()) * ratio), qRound(shape.top() * ratio),
              qRound((shape.y() + shape.height()) * ratio));
}

void CropShapeEditor::setCrop(const Crop &crop) {
  shape = cropToShape(crop);
  update();
  emit notifyCropSet(getCrop());
}

bool CropShapeEditor::isAutoCropSupported() const { return false; }

const QRectF CropShapeEditor::getCentralArea(const QRectF rectangle) const {
  return QRectF(rectangle.left() + SEL_OFFSET, rectangle.top() + SEL_OFFSET, rectangle.width() - SEL_OFFSET,
                rectangle.height() - SEL_OFFSET);
}

float CropShapeEditor::getRatio() const { return float(frameSize.width()) / thumbnailSize.width(); }
