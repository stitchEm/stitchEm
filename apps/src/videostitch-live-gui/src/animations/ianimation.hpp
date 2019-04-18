// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef IANIMATION_HPP
#define IANIMATION_HPP

#include <QWidget>
#include <QPropertyAnimation>

class IAnimation : public QPropertyAnimation {
 public:
  explicit IAnimation(QWidget* const widget);

  virtual void inAnimation() = 0;

  virtual void outAnimation() = 0;

  QWidget* internalWidget;
};

#endif  // IANIMATION_HPP
