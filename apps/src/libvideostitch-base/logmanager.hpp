// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "vslog.hpp"
#include "singleton.hpp"

#include <QFile>
#include <QTextStream>

namespace VideoStitch {
namespace Helper {
/**
 * @brief The LogManager class is a class dedicated to the log management
 *        It receives logs from VSLog, and filters them and send them to the gui
 */
class VS_COMMON_EXPORT LogManager : public QObject, public Singleton<LogManager> {
  friend class Singleton<LogManager>;
  Q_OBJECT
 public:
  /**
   * \enum Level
   * \brief Defines the log output levels.
   */
  enum class Level {
    Error,   /**< Error log level: unrecoverable error; the (part of) program will likely stop */
    Warning, /**< Warning log level: recoverable errors; the result will not be perfect (e.g. the input contains errors,
                etc.) */
    Info,    /**< Info log level: essential traces worthable reading */
    Verbose, /**< Verbose log level: extensive traces worthable reading */
    Debug    /**< Debug log level: debug traces, mainly for developers */
  };
  static QString getStringFromLevel(Level level);

  /**
   * @brief Start ups the logger
   */
  void setUpLogger();
  /**
   * @brief Returns the error log
   * @return The error log
   */
  VSLog* getErrorLog();
  /**
   * @brief Returns the error log
   * @return The error log
   */
  VSLog* getWarningLog();
  /**
   * @brief Returns the warning log
   * @return The warning log
   */
  VSLog* getInfoLog();
  /**
   * @brief Returns the info log
   * @return The info log
   */
  VSLog* getVerboseLog();
  /**
   * @brief Returns the verbose log
   * @return The verbose log
   */
  VSLog* getDebugLog();

  QList<Level> getLevels() const;
  const char* getSignalForLevel(Level level) const;

 public slots:
  void writeToLogFile(const QString& logMessage);
  /**
   * @brief Slot called to close the logger when the application is about to Quit
   */
  void close();

 signals:
  void reqWriteToLogFile(const QString& logMsg);
  void newError(const QString& error);
  void newWarning(const QString& error);
  void newDebug(const QString& error);
  void newInfo(const QString& error);
  void newVerbose(const QString& error);

 private slots:
  /**
   * @brief Write a log message to the log file.
   * @param Log message to write
   */
  void writeToLogFilePrivate(const QString& logMsg);
  void addDate(const QString& inputLogMessage);

 private:
  explicit LogManager(QObject* parent = 0);
  ~LogManager();

  /**
   * @brief Configures the error log sink
   */
  void configureErrorLog();
  /**
   * @brief Configures the warning log sink
   */
  void configureWarningLog();
  /**
   * @brief Configures the info log sink
   */
  void configureInfoLog();
  /**
   * @brief Configures the verbose log sink
   */
  void configureVerboseLog();
  /**
   * @brief Configures the debug log sink
   */
  void configureDebugLog();
  /**
   * @brief Configures the command log sink
   */
  void configureCmdLog();

  /**
   * Log file stuff
   */
  QString vsLogPath;
  QString vsLogSubDir;
  QScopedPointer<QFile> logFile;
  QScopedPointer<QTextStream> logDataStream;

  VSLog errLog, warnLog, infoLog, verbLog, debugLog, commandLog;
  QMap<Level, const char*> levelToSignal;
};
}  // namespace Helper
}  // namespace VideoStitch

Q_DECLARE_METATYPE(VideoStitch::Helper::LogManager::Level)
