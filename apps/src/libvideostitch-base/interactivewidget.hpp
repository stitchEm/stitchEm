// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "interactiverenderer.hpp"
#include "texture.hpp"

#include <QOpenGLWidget>
#include <QMouseEvent>
#include <QThread>
#include <QTimer>

#include <condition_variable>
#include <mutex>

#include <cmath>

/**
 * @brief The InteractiveWidget class
 *
 * A QOpenGLWidget wrapper around the interactive renderer.
 * Handles mouse event to set the viewport and the fov of the viewer.
 */
class VS_COMMON_EXPORT InteractiveWidget : public QOpenGLWidget {
  Q_OBJECT
 public:
  explicit InteractiveWidget(QWidget* parent = nullptr);
  virtual ~InteractiveWidget();

  Renderer& getRenderer();

  void initializeGL();
  void paintGL();
  void resizeGL(int width, int height);

  void mousePressEvent(QMouseEvent*);
  void mouseMoveEvent(QMouseEvent*);
  void wheelEvent(QWheelEvent*);
  void keyPressEvent(QKeyEvent*);
  void mouseDoubleClickEvent(QMouseEvent*);

  const QImage getScreenshot() { return grabFramebuffer(); }
 public slots:
  void setOrientation(double yaw, double pitch, double roll);

 signals:
  void reqFullscreen();

 private:
  QVector2D mousePressPosition;

  InteractiveRenderer renderer;

  QTimer updateTimer;
};
