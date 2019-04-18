// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "cropshapeeditor.hpp"

class VS_GUI_EXPORT CropCircleEditor : public CropShapeEditor {
  Q_OBJECT
 public:
  CropCircleEditor(const QSize thumbnailSize, const QSize frameSize, const Crop& initCrop,
                   QWidget* const parent = nullptr);
  virtual bool isValidCrop() const override;
  virtual bool isAutoCropSupported() const override;
  void setDefaultCircle();

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
  virtual void mouseMoveEvent(QMouseEvent* event) override;
  virtual void mouseReleaseEvent(QMouseEvent* event) override;

 private:
  // returns true if the point is in the border of the circle
  bool pointInBorder(const QRectF& rectangle, const QPoint& point) const;
  bool pointInCircle(const QRectF& rectangle, const QPoint& point) const;
  void setCursorWhenMoving(const QPoint& cursorPos);

  bool pointInTopLeftArea(const QRectF& rectangle, const QPoint& point) const;
  bool pointInTopRightArea(const QRectF& rectangle, const QPoint& point) const;
  bool pointInBottomLeftArea(const QRectF& rectangle, const QPoint& point) const;
  bool pointInBottomRightArea(const QRectF& rectangle, const QPoint& point) const;
  bool haveToResizeCircle;
  virtual void drawCropShape(QPainter& painter) override;
  void findAutoCrop();
  QImage cachedScaledFrame;
  QFuture<void> asyncAutoCropTask;
  QFutureWatcher<void> asyncAutoCropTaskWatcher;
};
