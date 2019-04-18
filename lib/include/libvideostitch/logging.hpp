// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LOGGING_HPP_
#define LOGGING_HPP_

#include "config.hpp"

#include <iosfwd>
#include <array>
#include <mutex>
#include <iostream>
#include <memory>
#include <assert.h>
#include <set>
#include <string>

namespace VideoStitch {
class Status;

/**
 * @brief A wrapper around std::ostream that blocks simultaneous access from multiple threads
 * The VideoStitch Logger can be accessed from multiple threads at once. The underyling std::ostream
 * is not thread-safe, and raw access to it can result in data races or undefined behavior.
 * For thread-safe usage, the Logger exposes this thread-safe variant.
 */
class VS_EXPORT ThreadSafeOstream {
 public:
  /**
   * Apply an I/O manipulator (like std::endl) to this thread-safe ostream.
   * Access is blocking.
   */
  ThreadSafeOstream& operator<<(std::ostream& (*manipulator)(std::ostream& os)) {
    if (ostream) {
      std::lock_guard<std::mutex> lock(*streamMutex);
      *ostream << manipulator;
    }
    return *this;
  }

  /**
   * Write to this thread-safe ostream.
   * Access is blocked while another thread uses this interface.
   */
  template <typename T>
  ThreadSafeOstream& operator<<(const T& v) {
    if (ostream) {
      std::lock_guard<std::mutex> lock(*streamMutex);
      *ostream << v;
    }
    return *this;
  }

  friend class Logger;

 private:
  ThreadSafeOstream(std::ostream* ostream, std::mutex* mutex) : streamMutex(mutex), ostream(ostream){};

  std::mutex* streamMutex;
  std::ostream* ostream;
};

#ifdef _MSC_VER
template class VS_EXPORT std::array<ThreadSafeOstream, 6>;
template class VS_EXPORT std::array<std::set<std::string>, 6>;
template class VS_EXPORT std::unique_ptr<std::mutex>;
#endif

/**
 * @brief Utility class for logging.
 */
class VS_EXPORT Logger {
 public:
  /**
   * \enum LogLevel
   * \brief Defines the logger output levels.
   */
  enum LogLevel {
    Quiet = -1,  /**< Quiet log level: disabled log (default: no output) */
    Error = 0,   /**< Error log level: unrecoverable error; the (part of) program will likely stop (default: stderr)*/
    Warning = 1, /**< Warning log level: recoverable errors; the result will not be perfect (e.g. the input contains
                    errors, etc.) (default: stderr) */
    Info = 2,    /**< Info log level: essential traces worthable reading (default: stdout) */
    Verbose = 3, /**< Verbose log level: extensive traces worthable reading (default: stdout) */
    Debug = 4    /**< Debug log level: debug traces, mainly for developers (default: stdout) */
  };

  /**
   * Set the global log level.
   * @note Not thread-safe.
   */
  static void setLevel(LogLevel level);

  /**
   * Set the global log level by reading argv, and remove it from the list of arguments.
   * Format is '-v 0' for Error to '-v 4' for debug. '-v q' sets to quiet.
   * @note Not thread-safe.
   */
  static void readLevelFromArgv(int& argc, char** argv);

  /**
   * Get the global log level.
   * @note Not thread-safe.
   */
  static LogLevel getLevel();

  /**
   * Set the log stream. See LogLevel definition for default values.
   * @param level The level for which to set the output stream.
   * @param os The stream to use. Must not be NULL.
   * @note Not thread safe. Several levels can use the same
   * stream. Forwarded to default instance.
   */
  static void setLogStream(LogLevel level, std::ostream* os);

  /**
   * Get a log stream.
   * @param level The log level to use.
   * @return an ostream to be used for logging.
   * @note Forwarded to default instance.
   */
  static ThreadSafeOstream& get(LogLevel level);

  /**
   * Get a filtered log stream.
   * @param level The log level to use.
   * @param tags Tags for this log
   * @return an ostream to be used for logging if no filter is set or one of the tags matches a filter,
   *         a null stream otherwise.
   * @note Forwarded to default instance.
   */
  template <typename... Tags>
  static ThreadSafeOstream& get(LogLevel level, const Tags&... tags) {
    Logger* instance = getInstance();
    if (level <= getLevel()) {
      bool filtered = isFiltered(level, tags...);
      if (!filtered) {
        return outputTags(instance->getI(level), tags...);
      }
    }
    return instance->getI(Quiet);
  }

  // Helpers

  template <typename... Tags>
  static ThreadSafeOstream& error(const Tags&... tags) {
    return get(Error, tags...);
  }

  template <typename... Tags>
  static ThreadSafeOstream& warning(const Tags&... tags) {
    return get(Warning, tags...);
  }

  template <typename... Tags>
  static ThreadSafeOstream& info(const Tags&... tags) {
    return get(Info, tags...);
  }

  template <typename... Tags>
  static ThreadSafeOstream& verbose(const Tags&... tags) {
    return get(Verbose, tags...);
  }

  template <typename... Tags>
  static ThreadSafeOstream& debug(const Tags&... tags) {
    return get(Debug, tags...);
  }

  /*
   * Concatenate tags in a readable way
   */
  static std::string concatenateTags(const std::string& first) { return "[" + first + "] "; }

  template <typename... Tags>
  static std::string concatenateTags(const std::string& first, const Tags&... others) {
    return concatenateTags(first) + concatenateTags(others...);
  }

  /**
   * Adds a filter for a log level.
   * @param level Target log level.
   * @param filter Filter
   * @note When at least one filter is set for a log level, all the messages are filtered out except those sent to a
   * stream retrieved with a matching tag.
   */
  static void addTagFilter(LogLevel level, const std::string& filter);

  /**
   * Removes a filter for a log level.
   * @param level Target log level.
   * @param filter Filter
   */
  static void removeTagFilter(LogLevel level, const std::string& filter);

  /**
   * Sets default values for log streams.
   */
  static void setDefaultStreams() { getInstance()->setDefaultStreamsI(); }

  Logger();

 private:
  static ThreadSafeOstream& outputTags(ThreadSafeOstream& out, const std::string& first) {
    return out << "[" << first << "] ";
  }

  template <typename... Tags>
  static ThreadSafeOstream& outputTags(ThreadSafeOstream& out, const std::string& first, const Tags&... others) {
    outputTags(out, first);
    outputTags(out, others...);
    return out;
  }

  static Logger* getInstance();

  void setLogStreamI(LogLevel level, std::ostream* os);
  ThreadSafeOstream& getI(LogLevel level);

  void setDefaultStreamsI();

  static bool isFiltered(LogLevel level, const std::string& first);

  template <typename... Tags>
  static bool isFiltered(LogLevel level, const std::string& first, const Tags&... others) {
    return isFiltered(level, first) || isFiltered(level, others...);
  }

  std::unique_ptr<std::mutex> mutex;
  std::array<ThreadSafeOstream, 6> streams;
  std::array<std::set<std::string>, 6> filters;
};

}  // namespace VideoStitch

#endif
