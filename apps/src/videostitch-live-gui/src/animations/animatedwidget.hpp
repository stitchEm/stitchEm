// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef ANIMATEDWIDGET_HPP
#define ANIMATEDWIDGET_HPP

#include <QWidget>
#include <QVector>
#include "ianimation.hpp"

class AnimatedWidget {
 public:
  AnimatedWidget();

  ~AnimatedWidget();

  void installAnimation(IAnimation* const animation);

  void startArrivalAnimations();

  void startDepartureAnimations();

 private:
  QVector<IAnimation*> animations;
};

#endif  // ANIMATEDWIDGET_HPP
