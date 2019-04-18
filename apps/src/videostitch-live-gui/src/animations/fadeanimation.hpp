// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef FADEANIMATION_HPP
#define FADEANIMATION_HPP

#include "ianimation.hpp"

class FadeAnimation : public IAnimation {
 public:
  explicit FadeAnimation(QWidget* const widget);

  virtual void inAnimation();

  virtual void outAnimation();
};

#endif  // FADEANIMATION_HPP
