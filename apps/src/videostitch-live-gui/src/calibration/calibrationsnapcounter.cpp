// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationsnapcounter.hpp"
#include "guiconstants.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QTimer>

static const int COUNTER_WIDTH(400);
static const int COUNTER_HEIGHT(200);
static const int ONE_SECOND(1000);

CalibrationSnapCounter::CalibrationSnapCounter(QWidget* const parent)
    : QFrame(parent), timer(new QTimer(this)), labelCounterValue(new QLabel(this)), counterValue(0), ticNumber(0) {
  labelCounterValue->setAlignment(Qt::AlignCenter);
  labelCounterValue->setObjectName("labelCounterValue");
  labelCounterValue->setFixedSize(COUNTER_WIDTH, COUNTER_HEIGHT);

  QHBoxLayout* layoutVertical = new QHBoxLayout(this);
  layoutVertical->addWidget(labelCounterValue);
  setGeometry(0, 0, parent->width(), parent->height());

  connect(timer, &QTimer::timeout, this, &CalibrationSnapCounter::onTimerUpdate);
}

void CalibrationSnapCounter::startCounter(const int value) {
  setFocus();
  ticNumber = 0;
  counterValue = value;
  labelCounterValue->setText(QString::number(value));
  timer->start(ONE_SECOND);
}

void CalibrationSnapCounter::onTimerUpdate() {
  ++ticNumber;
  if (ticNumber < counterValue) {
    labelCounterValue->setText(QString::number(counterValue - ticNumber));
  } else {
    labelCounterValue->setText(tr("Snap!"));
    timer->stop();
    emit notifyTimerEnded();
    close();
  }
}
