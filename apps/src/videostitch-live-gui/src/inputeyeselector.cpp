// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "inputeyeselector.hpp"

InputEyeSelector::InputEyeSelector(const QString name, const bool left, const bool right, QWidget* const parent)
    : QFrame(parent) {
  setupUi(this);
  labelInputName->setText(name);
  buttonLeft->setChecked(left);
  buttonRight->setChecked(right);
}

InputEyeSelector::~InputEyeSelector() {}

bool InputEyeSelector::isLeftEye() const { return buttonLeft->isChecked(); }

bool InputEyeSelector::isRightEye() const { return buttonLeft->isChecked(); }
