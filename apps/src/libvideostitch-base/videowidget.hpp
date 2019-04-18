// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "common-config.hpp"
#include "projection.hpp"
#include "texture.hpp"

#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QColor>
#include <QMouseEvent>
#include <QThread>
#include <QTimer>

#include <condition_variable>
#include <memory>
#include <mutex>

class YPRSignalCaps;

static const float MAX_ZOOM(400.0f);  // 400% zoom
static const float MIN_ZOOM(100.0f);  // 100% zoom
static const unsigned int WHEEL_STEP(10);

class VS_COMMON_EXPORT VideoWidget : public QOpenGLWidget, public QOpenGLFunctions {
  Q_OBJECT

  Q_PROPERTY(QColor gridColor MEMBER gridColor DESIGNABLE true)
  Q_PROPERTY(QColor gridColorHighlight MEMBER gridColorHighlight DESIGNABLE true)
  Q_PROPERTY(int gridSizeX MEMBER gridSizeX DESIGNABLE true)
  Q_PROPERTY(int gridSizeY MEMBER gridSizeY DESIGNABLE true)

 public:
  explicit VideoWidget(QWidget* parent = nullptr);
  virtual ~VideoWidget();

  void initializeGL();
  void paintGL();
  void resizeGL(int w, int h);

  void setZoomActivated(bool active);
  void restoreZoom();

  const QImage getScreenshot() { return QOpenGLWidget::grabFramebuffer(); }

 public slots:
  void mousePressEvent(QMouseEvent* e);
  void mouseMoveEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent*);
  void wheelEvent(QWheelEvent* event);
  void keyPressEvent(QKeyEvent* event);

  void setEditOrientationActivated(bool active);
  void setProjection(VideoStitch::Projection p, double hfov);

 signals:
  void rotatePanorama(YPRSignalCaps*);
  void applyOrientation();

 private:
  void point2pano(QPointF& p);
  float getTextureAspectRatio() const;
  float getWidgetAspectRatio() const;
  float sphereRadius();

 private:
  void paintPano();
  void paintCubemap();
  void paintDice();
  void paintCompact();
  void paintGrid();

  YPRSignalCaps* yprsignal;
  QPointF curPos, prevPos;
  double yaw, pitch, roll;
  Qt::MouseButton button;

  VideoStitch::Projection proj;
  double HFOV;
  float zoom;
  QPointF pan;
  bool editOrientation;
  bool enableZoom;

  int width_, height_;

  QColor gridColor, gridColorHighlight;
  int gridSizeX, gridSizeY;
};
