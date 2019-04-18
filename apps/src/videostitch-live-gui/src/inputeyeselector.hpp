// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef INPUTEYESELECTOR_HPP
#define INPUTEYESELECTOR_HPP

#include <QWidget>
#include "ui_inputeyeselector.h"

class InputEyeSelector : public QFrame, public Ui::InputEyeSelectorClass {
  Q_OBJECT

 public:
  explicit InputEyeSelector(const QString name, const bool left, const bool right, QWidget* const parent = nullptr);
  ~InputEyeSelector();
  bool isLeftEye() const;
  bool isRightEye() const;
};

#endif  // INPUTEYESELECTOR_HPP
