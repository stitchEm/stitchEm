// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QFrame>

namespace Ui {
class SequenceInfo;
}

class TimeSequenceWidget : public QFrame {
  Q_OBJECT

 public:
  explicit TimeSequenceWidget(QWidget *parent = 0);
  ~TimeSequenceWidget();

 public slots:
  void sequenceUpdated(const QString start, const QString stop);

 private:
  Ui::SequenceInfo *ui;
};
