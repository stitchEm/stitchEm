// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "animatedwidget.hpp"

AnimatedWidget::AnimatedWidget() {}

AnimatedWidget::~AnimatedWidget() {
  for (IAnimation* anim : animations) {
    delete anim;
  }
  animations.clear();
}

void AnimatedWidget::installAnimation(IAnimation* const animation) { animations.push_back(animation); }

void AnimatedWidget::startArrivalAnimations() {
  for (IAnimation* anim : animations) {
    anim->inAnimation();
  }
}

void AnimatedWidget::startDepartureAnimations() {
  for (IAnimation* anim : animations) {
    anim->inAnimation();
  }
}
