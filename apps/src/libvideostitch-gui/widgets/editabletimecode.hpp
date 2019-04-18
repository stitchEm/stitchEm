// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QLineEdit>

/**
 * @brief The EditableTimeCode class is a derivate from QLineEdit which is used to display Timecodes.
 *        It holds a QValidator which validates the text to check if the timecode is illformed.
 */
class VS_GUI_EXPORT EditableTimeCode : public QLineEdit {
  Q_OBJECT
 public:
  explicit EditableTimeCode(QWidget *parent = nullptr);
  void updateInputMask(bool moreThanOneHour, bool threeDigitsFps);

 public slots:
  void setText(const QString &text);

 private:
  virtual void focusOutEvent(QFocusEvent *) override;
  bool moreThanOneHour;
};
