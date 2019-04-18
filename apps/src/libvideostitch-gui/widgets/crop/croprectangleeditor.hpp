// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "cropshapeeditor.hpp"

class VS_GUI_EXPORT CropRectangleEditor : public CropShapeEditor {
  Q_OBJECT
 public:
  CropRectangleEditor(const QSize thumbnailSize, const QSize frameSize, const Crop& initCrop,
                      QWidget* const parent = nullptr);

 public slots:
  virtual void setDefaultCrop() override;

 protected:
  virtual const QRectF getTopArea(const QRectF rectangle) const override;
  virtual const QRectF getBottomArea(const QRectF rectangle) const override;
  virtual const QRectF getLeftArea(const QRectF rectangle) const override;
  virtual const QRectF getRightArea(const QRectF rectangle) const override;
  virtual const QRectF getTopLeftCorner(const QRectF rectangle) const override;
  virtual const QRectF getTopRightCorner(const QRectF rectangle) const override;
  virtual const QRectF getBottomLeftCorner(const QRectF rectangle) const override;
  virtual const QRectF getBottomRightCorner(const QRectF rectangle) const override;
  virtual void mousePressEvent(QMouseEvent* event) override;
  virtual void mouseReleaseEvent(QMouseEvent* event) override;
  virtual void mouseMoveEvent(QMouseEvent* event) override;

 private:
  virtual void drawCropShape(QPainter& painter) override;
};
