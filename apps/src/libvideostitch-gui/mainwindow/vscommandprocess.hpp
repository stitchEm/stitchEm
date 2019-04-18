// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QObject>
#include <QProcess>

#include <ostream>
#include <sstream>
#include <iostream>

class VS_GUI_EXPORT VSCommandProcess : public QObject {
  Q_OBJECT
 public:
  explicit VSCommandProcess(QObject *parent = 0);
  QProcess::ProcessState processState() const;
 signals:
  void signalProgression(int current);
  void signalProgressionMessage(QString message);
  void signalProcessStateChanged(QProcess::ProcessState newState);
  void logMessage(QString message);
  void error(QProcess::ProcessError error);
  void finished(int exitCode, QProcess::ExitStatus status);
 public slots:
  void start(QStringList arguments);
  /**
   * @brief Slot called when the process has finished
   * @param exitCode Exit code of the process
   * @param exitStatus Exit status of the process
   */
  void processFinished(int exitCode, QProcess::ExitStatus exitStatus);
  /**
   * @brief Slot called when the process started
   */
  void processStarted();
  /**
   * @brief Slot called when an error occured
   * @param error error code
   */
  void processError(QProcess::ProcessError error);
  /**
   * @brief Processes the standard output and the standard error
   */
  void processOuput();
  /**
   * @brief Slot called when the process state changes
   * @param newState
   */
  void processStateChanged(QProcess::ProcessState newState);
  /**
   * @brief Slot called to stop the process
   */
  void processStop();

 private:
  /**
   * @brief getNumberFromVSCmdOutput returns the number following a string.
   * @param begin The string we look for at the beginning of @s.
   * @param s The string to look into.
   * @param n The return value when available.
   * @return
   */
  void getFrameInfo(QString message, int &frame) const;
  QProcess *process;
  QString program;
  // Count the number of processed frames
  mutable int processedFramesCount;
  mutable int firstFrame;
};
