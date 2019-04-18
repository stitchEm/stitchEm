// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "editabletimecode.hpp"

static const char *timeFormatMask("99:99:99");

EditableTimeCode::EditableTimeCode(QWidget *parent) : QLineEdit(parent), moreThanOneHour(false) {
  setFocusPolicy(Qt::StrongFocus);
  setCursorMoveStyle(Qt::VisualMoveStyle);
}

void EditableTimeCode::updateInputMask(bool moreThanOneHour, bool threeDigitsFps) {
  this->moreThanOneHour = moreThanOneHour;
  QString inputMask = timeFormatMask;
  if (moreThanOneHour) {
    inputMask = inputMask.prepend("99:");
  }
  if (threeDigitsFps) {
    inputMask = inputMask.append("9");
  }
  setInputMask(inputMask);
}

void EditableTimeCode::focusOutEvent(QFocusEvent *event) {
  emit editingFinished();
  QLineEdit::focusOutEvent(event);
}

void EditableTimeCode::setText(const QString &text) {
  QString finalText = text;
  if (moreThanOneHour && finalText.split(":").size() == 3) {
    finalText = finalText.prepend("00:");
  }
  if (finalText.split(":")[0].size() == 1) {
    finalText = finalText.prepend("0");
  }
  QLineEdit::setText(finalText);
}
