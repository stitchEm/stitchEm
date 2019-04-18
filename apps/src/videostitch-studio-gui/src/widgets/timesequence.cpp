// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "timesequence.hpp"
#include "ui_timesequence.h"

TimeSequenceWidget::TimeSequenceWidget(QWidget *parent) : QFrame(parent), ui(new Ui::SequenceInfo) {
  ui->setupUi(this);
}

TimeSequenceWidget::~TimeSequenceWidget() { delete ui; }

void TimeSequenceWidget::sequenceUpdated(const QString start, const QString stop) {
  ui->labelStart->setText(start);
  ui->labelStop->setText(stop);
}
