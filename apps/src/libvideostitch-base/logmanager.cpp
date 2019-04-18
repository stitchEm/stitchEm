// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "logmanager.hpp"

#include "file.hpp"

#include "version.hpp"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>

namespace VideoStitch {
namespace Helper {

LogManager::LogManager(QObject *parent) : QObject(parent), Singleton<LogManager>() {
  connect(this, SIGNAL(reqWriteToLogFile(QString)), this, SLOT(addDate(QString)), Qt::QueuedConnection);
  configureErrorLog();
  configureWarningLog();
  configureInfoLog();
  configureVerboseLog();
  configureDebugLog();
  configureCmdLog();
}

LogManager::~LogManager() {}

QString LogManager::getStringFromLevel(LogManager::Level level) {
  switch (level) {
    case Level::Error:
      return tr("Error");
    case Level::Warning:
      return tr("Warning");
    case Level::Info:
      return tr("Info");
    case Level::Verbose:
      return tr("Verbose");
    case Level::Debug:
      return tr("Debug");
    default:
      Q_ASSERT(false);
      return QString();
  }
}

void LogManager::setUpLogger() {
  QString vsDataFolder = File::getAppDataFolder();
  QDir().mkpath(vsDataFolder);
  QString logFilePath = QString("%0%1.log").arg(vsDataFolder).arg(QCoreApplication::applicationName());
  logFile.reset(new QFile(logFilePath));
  logFile->open(QIODevice::Append | QIODevice::Text);
  logDataStream.reset(new QTextStream(logFile.data()));

  emit reqWriteToLogFile("------------------------------------------------");
  emit reqWriteToLogFile("Opening " + QCoreApplication::applicationName());
  emit reqWriteToLogFile("App version: " + QCoreApplication::applicationVersion());
  emit reqWriteToLogFile(QString("Lib version: %0").arg(LIB_VIDEOSTITCH_VERSION));

  levelToSignal[Level::Error] = SIGNAL(newError(QString));
  levelToSignal[Level::Warning] = SIGNAL(newWarning(QString));
  levelToSignal[Level::Debug] = SIGNAL(newDebug(QString));
  levelToSignal[Level::Info] = SIGNAL(newInfo(QString));
  levelToSignal[Level::Verbose] = SIGNAL(newVerbose(QString));
}

void LogManager::configureErrorLog() {
  errLog.setObjectName("Error Log");
  errLog.setParent(this);
  connect(&errLog, SIGNAL(emitError(QString)), this, SLOT(addDate(QString)), Qt::QueuedConnection);
  connect(this, SIGNAL(newError(QString)), this, SLOT(writeToLogFilePrivate(QString)), Qt::QueuedConnection);
}

void LogManager::configureWarningLog() {
  warnLog.setObjectName("Warning Log");
  warnLog.setParent(this);
  connect(&warnLog, SIGNAL(emitError(QString)), this, SLOT(addDate(QString)), Qt::QueuedConnection);
  connect(this, SIGNAL(newWarning(QString)), this, SLOT(writeToLogFilePrivate(QString)), Qt::QueuedConnection);
}

void LogManager::configureInfoLog() {
  infoLog.setObjectName("Info Log");
  infoLog.setParent(this);
  connect(&infoLog, SIGNAL(emitError(QString)), this, SLOT(addDate(QString)), Qt::QueuedConnection);
  connect(this, SIGNAL(newInfo(QString)), this, SLOT(writeToLogFilePrivate(QString)), Qt::QueuedConnection);
}

void LogManager::configureVerboseLog() {
  verbLog.setObjectName("Verbose Log");
  verbLog.setParent(this);
  connect(&verbLog, SIGNAL(emitError(QString)), this, SLOT(addDate(QString)), Qt::QueuedConnection);
  connect(this, SIGNAL(newVerbose(QString)), this, SLOT(writeToLogFilePrivate(QString)), Qt::QueuedConnection);
}

void LogManager::configureDebugLog() {
  debugLog.setObjectName("Debug Log");
  debugLog.setParent(this);
  connect(&debugLog, SIGNAL(emitError(QString)), this, SLOT(addDate(QString)), Qt::QueuedConnection);
  connect(this, SIGNAL(newDebug(QString)), this, SLOT(writeToLogFilePrivate(QString)), Qt::QueuedConnection);
}

void LogManager::configureCmdLog() {
  commandLog.setObjectName("Command Log");
  commandLog.setParent(this);
  connect(&commandLog, SIGNAL(emitError(QString)), this, SLOT(writeToLogFilePrivate(QString)), Qt::QueuedConnection);
}

VSLog *LogManager::getErrorLog() { return &errLog; }

VSLog *LogManager::getWarningLog() { return &warnLog; }

VSLog *LogManager::getInfoLog() { return &infoLog; }

VSLog *LogManager::getVerboseLog() { return &verbLog; }

VSLog *LogManager::getDebugLog() { return &debugLog; }

QList<LogManager::Level> LogManager::getLevels() const {
  return QList<LogManager::Level>() << Level::Error << Level::Warning << Level::Info << Level::Verbose << Level::Debug;
}

const char *LogManager::getSignalForLevel(LogManager::Level level) const { return levelToSignal.value(level); }

void LogManager::writeToLogFile(const QString &logMessage) { emit reqWriteToLogFile(logMessage); }

void LogManager::close() {
  logFile->close();
  LogManager::getInstance()->destroy();
}

void LogManager::writeToLogFilePrivate(const QString &logMsg) {
  Q_ASSERT(logDataStream != nullptr);
  *logDataStream << logMsg << "\n";
  logDataStream->flush();
}

void LogManager::addDate(const QString &inputLogMessage) {
  const QString logLine =
      "[" + QDateTime::currentDateTime().toLocalTime().toString("yyyy/MM/dd hh:mm:ss") + "] " + inputLogMessage;
  if (sender() == this) {
    writeToLogFilePrivate(logLine);
  } else if (sender() == &errLog) {
    emit newError(logLine);
  } else if (sender() == &warnLog) {
    emit newWarning(logLine);
  } else if (sender() == &debugLog) {
    emit newDebug(logLine);
  } else if (sender() == &infoLog) {
    emit newInfo(logLine);
  } else if (sender() == &verbLog) {
    emit newVerbose(logLine);
  }
}
}  // namespace Helper
}  // namespace VideoStitch
