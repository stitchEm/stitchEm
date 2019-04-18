// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "vscommandprocess.hpp"
#include <QApplication>
#include <QDir>

VSCommandProcess::VSCommandProcess(QObject *parent) : QObject(parent) {
  program = QApplication::applicationDirPath() + QDir::separator() + "videostitch-cmd";
#ifdef Q_OS_WIN
  program += ".exe";
#endif

  if (!QFile(program).exists()) {
    std::cerr << "Couldn't find the stitching tool. Expected at: " << program.toStdString() << std::endl;
  }

  process = new QProcess(this);
  process->setProcessChannelMode(QProcess::MergedChannels);

  connect(process, SIGNAL(readyReadStandardOutput()), this, SLOT(processOuput()));
  connect(process, SIGNAL(started()), this, SLOT(processStarted()));
  connect(process, SIGNAL(stateChanged(QProcess::ProcessState)), this,
          SLOT(processStateChanged(QProcess::ProcessState)));
  connect(process, SIGNAL(stateChanged(QProcess::ProcessState)), this,
          SIGNAL(signalProcessStateChanged(QProcess::ProcessState)));
  connect(process, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(processFinished(int, QProcess::ExitStatus)));
  connect(process, SIGNAL(finished(int, QProcess::ExitStatus)), this, SIGNAL(finished(int, QProcess::ExitStatus)));
  connect(process, SIGNAL(error(QProcess::ProcessError)), this, SLOT(processError(QProcess::ProcessError)));
  connect(process, SIGNAL(error(QProcess::ProcessError)), this, SIGNAL(error(QProcess::ProcessError)));
}

QProcess::ProcessState VSCommandProcess::processState() const { return process->state(); }

void VSCommandProcess::start(QStringList arguments) {
  firstFrame = 0;
  processedFramesCount = 0;
  for (int i = 0; i < arguments.size() - 1; i++)
    if (arguments[i] == "-f") {
      firstFrame = arguments[i + 1].toInt();
    }
  process->start(program, arguments);
}

void VSCommandProcess::processOuput() {
  while (process->canReadLine()) {
    QByteArray error = process->readLine(256);
    int frame = -1;
    emit logMessage(QString(error).simplified());
    getFrameInfo(QString(error), frame);
    if (frame != -1) {
      emit signalProgressionMessage(tr("Processing frame %0").arg(QString::number(frame)));
      emit signalProgression(frame);
    }
  }
}

void VSCommandProcess::getFrameInfo(QString message, int &frame) const {
  if (message.startsWith("stitched frame")) {
    frame = processedFramesCount + firstFrame;
    processedFramesCount++;
  }
}

#ifdef _MSC_VER
#include <Windows.h>
namespace {
BOOL WINAPI signalHandler(_In_ DWORD dwCtrlType) { return dwCtrlType == CTRL_BREAK_EVENT; }
}  // namespace
#endif

void VSCommandProcess::processStop() {
#ifdef _MSC_VER
  {
    Q_PID pid = process->pid();
    if (pid) {
      if (!AttachConsole(pid->dwProcessId) || !SetConsoleCtrlHandler(signalHandler, TRUE) ||
          !GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, 0)) {
        std::cerr << "Couldn't attach to CSRSS: videostitch-cmd will likely be killed." << std::endl;
      }
    }
#else
  if ((process->state() == QProcess::Running) && !process->waitForFinished(1000)) {
    process->terminate();
#endif
    if (process->state() != QProcess::NotRunning) {
      emit signalProgressionMessage(tr("Waiting for the rendering process to complete."));
    }
    if ((process->state() == QProcess::Running) && !process->waitForFinished(3000)) {
      emit signalProgressionMessage(tr("The rendering process was canceled. The output might be incomplete."));
      process->kill();
    } else if (process->exitStatus() != QProcess::NormalExit) {
      emit signalProgressionMessage(
          tr("The rendering process has exited with error(s). Please check the output file integrity."));
    }
#ifdef _MSC_VER
    SetConsoleCtrlHandler(signalHandler, FALSE);
    FreeConsole();
#endif
  }
}

void VSCommandProcess::processStarted() { emit logMessage(tr("Process started.")); }

void VSCommandProcess::processError(QProcess::ProcessError error) {
  switch (error) {
    case QProcess::FailedToStart:
      emit logMessage(tr("Process failed to start."));
      break;
    case QProcess::Crashed:
      emit logMessage(tr("Process crashed."));
      break;
    case QProcess::Timedout:
      emit logMessage(tr("Process timed out."));
      break;
    case QProcess::WriteError:
      emit logMessage(tr("Process write error."));
      break;
    case QProcess::ReadError:
      emit logMessage(tr("Process read error."));
      break;
    case QProcess::UnknownError:
      emit logMessage(tr("Unknown error."));
      break;
  }
}

void VSCommandProcess::processStateChanged(QProcess::ProcessState newState) {
  switch (newState) {
    case QProcess::NotRunning:
      emit logMessage(tr("Process isn't running."));
      break;
    case QProcess::Running:
      emit logMessage(tr("Process is now running."));
      break;
    case QProcess::Starting:
      emit logMessage(tr("Process is starting."));
      break;
  }
}

void VSCommandProcess::processFinished(int exitCode, QProcess::ExitStatus exitStatus) {
  QString message;
  switch (exitStatus) {
    case QProcess::NormalExit:
      message = tr("Process exited normally.");
      break;
    case QProcess::CrashExit:
      message = tr("Process didn't exit normally.");
      break;
  }
  emit logMessage(message + tr(" Exit code %0").arg(exitCode));
}
