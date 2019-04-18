// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "batchertask.hpp"

#include <QApplication>
#include <QDir>
#include <QProgressBar>
#include <QStyle>
#include <sstream>

#include "autoelidelabel.hpp"
#include "libvideostitch-gui/mainwindow/vscommandprocess.hpp"
#include "libvideostitch-gui/utils/pluginshelpers.hpp"
#include "libvideostitch-base/logmanager.hpp"
#include "libvideostitch-base/common-config.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"

int BatcherTask::lastID = 0;

BatcherTask::BatcherTask(QObject *parent)
    : QObject(parent),
      ptvName(nullptr),
      taskProgress(nullptr),
      process(nullptr),
      state(Idle),
      id(-1),
      firstFrame(-1),
      lastFrame(-1) {}

bool BatcherTask::operator<(const BatcherTask &toCompare) { return id < toCompare.id; }

bool BatcherTask::operator>(const BatcherTask &toCompare) { return id > toCompare.id; }

void BatcherTask::setPtvName(AutoElideLabel *name) { ptvName = name; }

void BatcherTask::setID(int id) { this->id = id; }

QString BatcherTask::getLog() const { return log; }

QString BatcherTask::getPtvName() const { return ptvName->text(); }

int BatcherTask::getID() const { return id; }

BatcherTask::State BatcherTask::getState() const { return state; }

void BatcherTask::resetState() {
  setState(BatcherTask::Idle);
  taskProgress->setFormat("");
  setErrorStyle(false);
  taskProgress->setMaximum(100);
  taskProgress->setValue(0);
}

bool BatcherTask::isRunning() const { return (process) ? process->processState() == QProcess::Running : false; }

void BatcherTask::setTaskProgress(QProgressBar *taskProgress) { this->taskProgress = taskProgress; }

void BatcherTask::start() {
  if (state != BatcherTask::Idle) {
    emit finished();
    return;
  }

  QStringList deviceStrings;
  for (int device : devices) {
    deviceStrings.append(QString::number(device));
  }
  RetrieveFrameBounds(firstFrame, lastFrame);

  QStringList arguments;
  if (firstFrame != -1 && lastFrame != -1) {
    arguments << "-f" << QString::number(firstFrame);
    arguments << "-l" << QString::number(lastFrame);
  }
  arguments << "-v" << QString::number(3);
  arguments << "-i" << ptvName->text();
  arguments << "-d" << deviceStrings.join(",");
  arguments << "-p" << VideoStitch::Plugin::getCorePluginFolderPath();
  auto path = VideoStitch::Plugin::getGpuCorePluginFolderPath();
  if (!path.isEmpty()) {
    arguments << "-p" << path;
  }

  process = new VSCommandProcess(this);
  connect(process, SIGNAL(signalProgression(int)), this, SLOT(setProgress(int)));
  connect(process, SIGNAL(signalProcessStateChanged(QProcess::ProcessState)), this,
          SLOT(processStateChanged(QProcess::ProcessState)));
  connect(process, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(processFinished(int, QProcess::ExitStatus)));
  connect(process, SIGNAL(logMessage(QString)), this, SLOT(logMessage(QString)));
  connect(process, SIGNAL(logMessage(QString)), VideoStitch::Helper::LogManager::getInstance(),
          SLOT(writeToLogFile(QString)));

  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile("Starting to process " + ptvName->text());
  log = QString();
  process->start(arguments);
}

void BatcherTask::kill() {
  setState(Canceled);
  taskProgress->setFormat(tr("Canceled"));
  setErrorStyle(false);
  if (process) {
    process->processStop();
  }
}

void BatcherTask::setDevices(const QList<int> &deviceIds) { devices = deviceIds; }

void BatcherTask::setProgress(int current) {
  if (firstFrame != -1 && lastFrame != -1) {
    taskProgress->setFormat(tr("Processing") + " %p%");
    taskProgress->setMaximum(lastFrame - firstFrame);
    taskProgress->setValue(current - firstFrame);
  } else {
    // We don't have information about the last stitchable frame so we show the current processed frame.
    // TODO FIXME: VSA-6002
    taskProgress->setFormat(tr("Processing frame ") + QString::number(current));
    taskProgress->setMaximum(100);
    taskProgress->setValue(100);
  }
}

void BatcherTask::processStateChanged(QProcess::ProcessState newState) {
  switch (newState) {
    case QProcess::NotRunning:
      setState(Finished);
      break;
    case QProcess::Running:
      setState(Processing);
      break;
    case QProcess::Starting:
      setState(Processing);
      break;
  }
}

void BatcherTask::processFinished(int exitCode, QProcess::ExitStatus status) {
  Q_UNUSED(status)
  QString preffix;
  if (state == Canceled) {
    preffix = tr("Canceled");
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile("Process canceled: " + ptvName->text());
  } else {
    if (exitCode != 0) {
      setState(Error);
      preffix = tr("Error");
      VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
          ptvName->text() + " finished on error code: " + QString::number(exitCode));
    } else {
      setState(Finished);
      preffix = tr("Finished");
      VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(ptvName->text() + " was successfully processed");
    }
  }
  setErrorStyle(exitCode != 0);
  emit finished();
  process->deleteLater();
  process = nullptr;
  taskProgress->setFormat(preffix);
  taskProgress->setMaximum(100);
  taskProgress->setValue(100);
}

void BatcherTask::setState(State newState) { state = newState; }

void BatcherTask::logMessage(const QString &logLine) {
  this->log += logLine + "\n";
  emit newLogLine(logLine);
}

QString BatcherTask::getStringFromState() {
  switch (state) {
    case Processing:
      return tr("Processing");
    case Idle:
      return tr("Idle");
    case Error:
      return tr("Error");
    case Finished:
      return tr("Finished");
    case Canceled:
      return tr("Canceled");
    default:
      return tr("Unknown state");
  }
}

void BatcherTask::RetrieveFrameBounds(int &firstFrame, int &lastFrame) const {
  VideoStitch::Potential<VideoStitch::Ptv::Parser> parser(VideoStitch::Ptv::Parser::create());
  if (!parser.ok()) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile("Could not initialize the ptv parser.");
    return;
  }
  if (!parser->parse(ptvName->text().toLocal8Bit().constData())) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        QString("Could not parse the ptv: %0").arg(parser->getErrorMessage().c_str()));
    return;
  }
  const VideoStitch::Ptv::Value &ptv = parser->getRoot();
  firstFrame = ptv.has("first_frame")->asInt();
  lastFrame = ptv.has("last_frame")->asInt();
  if (ptv.has("output")) {
    const VideoStitch::Ptv::Value &output = *ptv.has("output");
    if (!output.has("process_sequence")) {
      firstFrame = -1;
      lastFrame = -1;
    }
  }
}

void BatcherTask::setErrorStyle(bool error) {
  taskProgress->setProperty("error", error);
  taskProgress->style()->unpolish(taskProgress);
  taskProgress->style()->polish(taskProgress);
  taskProgress->update();
}
