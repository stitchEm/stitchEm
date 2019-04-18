// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "processprogressdialog.hpp"

#include "ui_processprogressdialog.h"
#include "mainwindow/mainwindow.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/vscommandprocess.hpp"

#include "libvideostitch-base/linkhelpers.hpp"

#include "libvideostitch/config.hpp"

ProcessProgressDialog::ProcessProgressDialog(QStringList args, qint64 firstFrame, qint64 lastFrame, QWidget *parent)
    : QDialog(parent, Qt::WindowTitleHint),
      ui(new Ui::ProcessProgressDialog),
      commandProcess(new VSCommandProcess(this)),
      commandArgs(args),
      firstFrame(firstFrame),
      lastFrame(lastFrame) {
  ui->setupUi(this);
  ui->logWidget->hide();
  ui->warningIconLabel->setPixmap(
      QPixmap::fromImage(QImage(WARNING_ICON)).scaled(32, 32, Qt::KeepAspectRatio, Qt::SmoothTransformation));
  ui->warningIconLabel->hide();
  ui->upperWidget->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  ui->upperWidget->setFixedSize(500, 150);
  ui->progressBar->setMinimum(0);
  ui->progressBar->setMaximum(lastFrame);
  ui->progressBar->setValue(0);
  ui->progressLabel->setText(tr("Launching process"));

  connect(commandProcess, &VSCommandProcess::logMessage, ui->logWidget, &LogWidget::logMessage);
  connect(commandProcess, &VSCommandProcess::finished, this, &ProcessProgressDialog::processFinished);
  connect(commandProcess, &VSCommandProcess::signalProgression, this, &ProcessProgressDialog::setProgress);
  connect(commandProcess, &VSCommandProcess::signalProgressionMessage, ui->progressLabel, &QLabel::setText);
  connect(ui->stopButton, &QPushButton::clicked, commandProcess, &VSCommandProcess::processStop);
  connect(ui->detailsButton, &QPushButton::clicked, this, &ProcessProgressDialog::showDetailsClicked);
  ui->upperWidget->layout()->removeWidget(ui->logWidget);
  adjustSize();
  ui->logWidget->hideControls();
  setFixedWidth(width());
}

ProcessProgressDialog::~ProcessProgressDialog() { commandProcess->processStop(); }

int ProcessProgressDialog::exec() {
  commandProcess->start(commandArgs);
  return QDialog::exec();
}

void ProcessProgressDialog::showWarning() {
#ifdef Q_OS_WIN
  if (ui->progressBar->value() == ui->progressBar->minimum()) {
    emit reqChangeProgressValue(1, 1);
  }
  emit reqChangeProgressState(TBPF_ERROR);
#endif
  ui->progressLabel->setText(
      tr("An error occured while processing. Please check the logs to get more information.<br> Check out %0 to get "
         "support.")
          .arg(formatLink(VIDEOSTITCH_SUPPORT_URL, QCoreApplication::applicationName() + "'s Knowledge Base")));
  ui->progressBar->hide();
  ui->warningIconLabel->show();
}

void ProcessProgressDialog::processFinished(int exitCode, QProcess::ExitStatus exitStatus) {
  switch (exitStatus) {
    case QProcess::NormalExit:
      if (exitCode == 0) {
        accept();
      } else {
        showWarning();
      }
      break;
    case QProcess::CrashExit:
      showWarning();
      break;
  }

  ui->stopButton->setText(tr("Close"));
  connect(ui->stopButton, &QPushButton::clicked, this, &ProcessProgressDialog::close);
}

void ProcessProgressDialog::showDetailsClicked(bool show) {
  ui->logWidget->setVisible(show);
  if (show) {
    ui->detailsButton->setText(tr("Hide log"));
    layout()->addWidget(ui->logWidget);
  } else {
    ui->detailsButton->setText(tr("Show log"));
    layout()->removeWidget(ui->logWidget);
  }
  adjustSize();
}

void ProcessProgressDialog::setProgress(int current) {
  ui->progressBar->setMaximum(lastFrame - firstFrame);
  ui->progressBar->setValue(current - firstFrame);
#ifdef Q_OS_WIN
  emit reqChangeProgressValue(current - firstFrame, lastFrame - firstFrame);
#endif
}

void ProcessProgressDialog::closeEvent(QCloseEvent *event) {
#ifdef Q_OS_WIN
  emit reqChangeProgressState(TBPF_NOPROGRESS);
#endif
  QDialog::closeEvent(event);
}
