// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ianimation.hpp"

IAnimation::IAnimation(QWidget* const widget) : QPropertyAnimation(new QPropertyAnimation()), internalWidget(widget) {}
