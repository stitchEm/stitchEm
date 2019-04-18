// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "timewidget.hpp"
#include "ui_timewidget.h"

#include "libvideostitch-gui/mainwindow/timeconverter.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"

#include <QKeyEvent>

TimeWidget::TimeWidget(QWidget *parent)
    : QWidget(parent), ui(new Ui::TimeWidget), editable(false), maxFrame(0), defaultFrame(0) {
  ui->setupUi(this);
  setEditable(false);
}

TimeWidget::~TimeWidget() { delete ui; }

bool TimeWidget::getTimecodeVisibility() const { return ui->timeEdit->isVisible(); }

bool TimeWidget::getFramenumberVisibility() const { return ui->frameEdit->isVisible(); }

void TimeWidget::setTimecodeVisibility(bool visible) { ui->timeEdit->setVisible(visible); }

void TimeWidget::setFramenumberVisibility(bool visible) { ui->frameEdit->setVisible(visible); }

void TimeWidget::setEditable(bool state) {
  editable = state;
  ui->timeEdit->setEnabled(editable);
}

bool TimeWidget::isEditable() const { return editable; }

void TimeWidget::reset() {
  ui->timeEdit->setText("00:00:00");
  ui->frameEdit->setText("0");
}

void TimeWidget::updateInputMask(bool moreThanOneHour, bool threeDigitsFps) {
  ui->timeEdit->updateInputMask(moreThanOneHour, threeDigitsFps);
}

void TimeWidget::setMaxFrame(const frameid_t max) { maxFrame = max; }

void TimeWidget::setFrame(frameid_t frame) {
  defaultFrame = frame;
  ui->frameEdit->setText(QString::number(frame));
  ui->timeEdit->setText(
      TimeConverter::frameToTimeDisplay(frame, GlobalController::getInstance().getController()->getFrameRate()));
}

void TimeWidget::mouseDoubleClickEvent(QMouseEvent *e) {
  if (!editable) {
    QWidget::mouseDoubleClickEvent(e);
    return;
  }
}

void TimeWidget::keyPressEvent(QKeyEvent *e) {
  if (!editable) {
    return;
  }

  if (e->key() == Qt::Key_Return || e->key() == Qt::Key_Enter) {
    bool success = false;
    frameid_t frameId = ui->frameEdit->text().toInt(&success);
    if (success) {
      emit frameChanged(frameId);
    }

    frameId = TimeConverter::timeDisplayToFrame(
        ui->timeEdit->text(), GlobalController::getInstance().getController()->getFrameRate(), &success);
    if (success) {
      emit frameChangedFromTimeCode(frameId > maxFrame ? defaultFrame : frameId);
    } else {
      emit frameChangedFromTimeCode(defaultFrame);
    }
  } else if (e->key() == Qt::Key_Escape) {
    emit frameChangedFromTimeCode(defaultFrame);
  }
}
