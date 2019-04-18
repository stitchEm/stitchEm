// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QGraphicsEffect>
#include "dialogbackground.hpp"
#include "guiconstants.hpp"

DialogBackground::DialogBackground(QWidget* const parent) : QFrame(parent) {
  if (parent != nullptr) updateSize(parent->width(), parent->height());

  QGraphicsBlurEffect* blur(new QGraphicsBlurEffect(this));
  blur->setBlurRadius(BLUR_FACTOR);
  setGraphicsEffect(blur);
}

void DialogBackground::updateSize(unsigned int parentWidth, unsigned int parentHeight) {
  setFixedSize(parentWidth, parentHeight);
}
