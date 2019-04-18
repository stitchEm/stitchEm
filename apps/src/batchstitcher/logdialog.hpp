// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LOGDIALOG_HPP
#define LOGDIALOG_HPP

#include <QDialog>

namespace Ui {
class LogDialog;
}

class LogDialog : public QDialog {
  Q_OBJECT

 public:
  explicit LogDialog(const QString &log, const QString &ptvName, QWidget *parent = 0);
  ~LogDialog();
 private slots:
  void appendLogLine(const QString &logLine);

 private:
  Ui::LogDialog *ui;
};

#endif  // LOGDIALOG_HPP
