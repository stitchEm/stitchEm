// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "logdialog.hpp"
#include "ui_logdialog.h"

LogDialog::LogDialog(const QString &log, const QString &ptvName, QWidget *parent)
    : QDialog(parent, Qt::WindowTitleHint), ui(new Ui::LogDialog) {
  ui->setupUi(this);
  setWindowTitle(ptvName + " " + tr("log"));
  // remove the last carriage return
  QString logWithoutLastReturn = log;
  logWithoutLastReturn.truncate(logWithoutLastReturn.size() - 1);
  ui->logField->append(logWithoutLastReturn);
}

LogDialog::~LogDialog() { delete ui; }

void LogDialog::appendLogLine(const QString &logLine) { ui->logField->append(logLine); }
