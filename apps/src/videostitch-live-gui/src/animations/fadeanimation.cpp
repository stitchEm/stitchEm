// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QWidget>
#include <QGraphicsOpacityEffect>
#include "fadeanimation.hpp"

FadeAnimation::FadeAnimation(QWidget* const widget) : IAnimation(widget) {
  setDuration(200);
  setPropertyName("opacity");

  QGraphicsOpacityEffect* backgroundOpacity = new QGraphicsOpacityEffect(this);
  internalWidget->setGraphicsEffect(backgroundOpacity);
  setTargetObject(backgroundOpacity);
  backgroundOpacity->setOpacity(0.0);
}

void FadeAnimation::inAnimation() {
  setStartValue(0.0);
  setEndValue(1.0);
  setEasingCurve(QEasingCurve::InOutQuad);
  start();
}

void FadeAnimation::outAnimation() {
  setStartValue(1.0);
  setEndValue(0.0);
  setEasingCurve(QEasingCurve::OutQuad);
  start();
}
