// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "batchwindow.hpp"
#include "ui_batchwindow.h"

#include "libvideostitch-gui/mainwindow/packet.hpp"
#include "libvideostitch-gui/mainwindow/processutility.hpp"

#include "libvideostitch-base/logmanager.hpp"

#include "libvideostitch/gpu_device.hpp"

#include "libgpudiscovery/genericDeviceInfo.hpp"

#include <QFileDialog>
#include <QTimer>
#include <QUrl>
#include <QDir>
#include <QMessageBox>
#include <QMimeData>
#include <QDropEvent>
#include <QProcess>
#include <QFileInfo>
#include <fstream>

static const QString STUDIO_COMMAND_NAME = "videostitch-studio";

BatchWindow::BatchWindow(const QString &fileToOpen, QWidget *parent)
    : QMainWindow(parent), ui(new Ui::BatchWindow), idle(new QState()), processing(new QState()) {
  ui->setupUi(this);
  setWindowTitle(QCoreApplication::applicationName());

  connect(ui->removeButton, SIGNAL(clicked()), this, SLOT(onRemoveAllButtonClicked()));
  connect(ui->tableWidget, SIGNAL(reqOpenVS(QString)), this, SLOT(openVS(QString)));
  connect(this, &BatchWindow::signalDeviceSelectionChanged, ui->tableWidget, &TaskTable::onDeviceSelectionChanged);
  connect(ui->deviceList, &GpuListWidget::selectedGpusChanged, this, &BatchWindow::onDeviceSelectionChanged);
  connect(ui->tableWidget, SIGNAL(removedTask()), this, SLOT(updateButtonStates()));

  stateMachine.addState(idle);
  stateMachine.addState(processing);
  idle->addTransition(ui->tableWidget, SIGNAL(processing()), processing);
  processing->addTransition(ui->tableWidget, SIGNAL(finished()), idle);
  stateMachine.setInitialState(idle);
  stateMachine.start();

  idle->assignProperty(ui->stitchButton, "text", tr("Stitch!"));
  processing->assignProperty(ui->stitchButton, "text", tr("Cancel"));
  connect(ui->actionExit, SIGNAL(triggered()), this, SLOT(close()));

  onDeviceSelectionChanged();
  if (!fileToOpen.isEmpty()) {
    ui->tableWidget->addTask(fileToOpen);
  }
  updateButtonStates();
}

BatchWindow::~BatchWindow() { delete ui; }

void BatchWindow::openVS(QString fileToOpen) {
  Q_ASSERT(vsProcess == nullptr);
  vsProcess = new QProcess(this);
  connect(vsProcess, SIGNAL(finished(int)), this, SLOT(vsProcessTerminated()));

  QString program = QApplication::applicationDirPath() + QDir::separator() + STUDIO_COMMAND_NAME;
#ifdef Q_OS_WIN
  program += ".exe";
#endif
  QStringList arguments;

  arguments << fileToOpen;
  vsProcess->startDetached(program, arguments);
}

void BatchWindow::vsProcessTerminated() { delete sender(); }

void BatchWindow::dropEvent(QDropEvent *e) {
  if (e->mimeData()->hasUrls()) {
    QList<QUrl> urlList = e->mimeData()->urls();
    if (!urlList.size()) {
      return;
    }
    foreach (QUrl url, urlList) {
      if (QFileInfo(url.toLocalFile()).isFile() &&
          (url.toString().endsWith(".ptv") || url.toString().endsWith(".ptvb"))) {
        ui->tableWidget->addTask(url.toLocalFile());
        updateButtonStates();
      }
    }
  }
}

void BatchWindow::dragMoveEvent(QDragMoveEvent *e) { e->accept(); }

void BatchWindow::dragEnterEvent(QDragEnterEvent *e) { e->acceptProposedAction(); }

void BatchWindow::processMessage(const Packet &packet) {
  switch (packet.getType()) {
    case Packet::WAKEUP:
      showMaximized();
      break;
    case Packet::OPEN_FILES: {
      QString argString = QString::fromLatin1(packet.getPayload());
      QStringList args = QStringList() << argString;
      foreach (QString file, args) {
        ui->tableWidget->addTask(file);
        updateButtonStates();
      }
      break;
    }
    default:
      break;
  }
}

void BatchWindow::closeEvent(QCloseEvent *event) {
  QMainWindow::closeEvent(event);
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile("Closing " + QCoreApplication::applicationName());
}

void BatchWindow::updateButtonStates() {
  const bool hasElements = ui->tableWidget->rowCount() > 0;
  ui->stitchButton->setEnabled(hasElements);
  ui->removeButton->setEnabled(hasElements);
  ui->actionRemove_Selected->setEnabled(hasElements);
}

void BatchWindow::onDeviceSelectionChanged() { emit signalDeviceSelectionChanged(getSelectedDevices()); }

QList<int> BatchWindow::getSelectedDevices() { return ui->deviceList->getSelectedGpus().toList(); }

void BatchWindow::on_actionOpen_Project_triggered() {
  QStringList projects = QFileDialog::getOpenFileNames(
      this, tr("Add projects to list"), "", tr("%0 Project(*.ptv *.ptvb)").arg(QCoreApplication::applicationName()));
  if (projects.isEmpty()) {
    return;
  }
  foreach (QString s, projects) {
    ui->tableWidget->addTask(s);
    updateButtonStates();
  }
}

void BatchWindow::on_actionRemove_Selected_triggered() {
  ui->tableWidget->removeSelected();
  updateButtonStates();
}

void BatchWindow::on_stitchButton_clicked() {
  if (ui->tableWidget->rowCount() == 0) {
    return;
  }
  if (ui->tableWidget->getState() != TaskTable::RUNNING) {
    QString vsProcessName = STUDIO_COMMAND_NAME;
#ifdef Q_OS_WIN
    vsProcessName += ".exe";
#endif
    if (ProcessUtility::getProcessByName(vsProcessName) > 0) {
      int ret = QMessageBox::warning(
          this, tr("Warning: %0 is running.").arg(VIDEOSTITCH_STUDIO_APP_NAME),
          tr("%0 is currently running.").arg(VIDEOSTITCH_STUDIO_APP_NAME) + QString("<br>") +
              tr("To avoid performance issues you should close %0.").arg(VIDEOSTITCH_STUDIO_APP_NAME),
          QMessageBox::Ok, QMessageBox::Cancel);
      if (ret != QMessageBox::Ok) {
        return;
      }
    }
  }

  ui->tableWidget->startStitching();
}

void BatchWindow::onRemoveAllButtonClicked() {
  ui->tableWidget->removeAll();
  updateButtonStates();
}
