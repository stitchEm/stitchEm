// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "interactivewidget.hpp"

InteractiveWidget::InteractiveWidget(QWidget* parent) : QOpenGLWidget(parent), renderer() {}

InteractiveWidget::~InteractiveWidget() {}

Renderer& InteractiveWidget::getRenderer() { return renderer; }

void InteractiveWidget::initializeGL() { renderer.initialize(); }
void InteractiveWidget::paintGL() { renderer.render(); }
void InteractiveWidget::resizeGL(int width, int height) { renderer.resize(width, height); }

#define DEFAULT_FOV 120
#define FOV_MAX 140
#define FOV_MIN 5

void InteractiveWidget::mousePressEvent(QMouseEvent* event) {
  mousePressPosition.setX(event->x());
  mousePressPosition.setY(event->y());
  update();
}

#define MOVEFACTOR 1.8
#define MOVEFORMULA(m) (((m)*renderer.fov * renderer.fov) / (DEFAULT_FOV * DEFAULT_FOV * MOVEFACTOR))
#define LL_EPSILON (M_PI / 1000.0)
void InteractiveWidget::mouseMoveEvent(QMouseEvent* event) {
  float dx = MOVEFORMULA(event->x() - mousePressPosition.x());
  float dy = MOVEFORMULA(event->y() - mousePressPosition.y());
  mousePressPosition.setX(event->x());
  mousePressPosition.setY(event->y());
  renderer.lngRot += dx / renderer.fov;
  renderer.latRot -= dy / renderer.fov;
  if (renderer.latRot >= M_PI / 2 - LL_EPSILON) {
    renderer.latRot = (float)(M_PI / 2 - LL_EPSILON);
  } else if (renderer.latRot <= -M_PI / 2 + LL_EPSILON) {
    renderer.latRot = (float)(-M_PI / 2 + LL_EPSILON);
  }
  update();
}
#undef LL_EPSILON

void InteractiveWidget::wheelEvent(QWheelEvent* event) {
  // delta is typically in 120 unit increments
  float m = pow(1.1, event->delta() / 120);
  renderer.fov = renderer.fov / m;
  if (renderer.fov > FOV_MAX) {
    renderer.fov = FOV_MAX;
  }
  if (renderer.fov < FOV_MIN) {
    renderer.fov = FOV_MIN;
  }
  renderer.updateZoom();
  update();
}

void InteractiveWidget::keyPressEvent(QKeyEvent* event) { event->ignore(); }

void InteractiveWidget::mouseDoubleClickEvent(QMouseEvent* e) {
  Q_UNUSED(e)
  emit reqFullscreen();
}

void InteractiveWidget::setOrientation(double yaw, double pitch, double roll) {
  renderer.lngRot = yaw;
  renderer.latRot = pitch;
  renderer.roll = roll;
}
