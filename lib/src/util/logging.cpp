// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/logging.hpp"

#include <map>
#include <streambuf>
#include <string>

namespace VideoStitch {

static Logger::LogLevel globalLogLevel = Logger::Info;

Logger::Logger()
    : mutex(new std::mutex),
      streams({{
          ThreadSafeOstream(nullptr, nullptr),
          ThreadSafeOstream(&std::cerr, mutex.get()),
          ThreadSafeOstream(&std::cerr, mutex.get()),
          ThreadSafeOstream(&std::cout, mutex.get()),
          ThreadSafeOstream(&std::cout, mutex.get()),
          ThreadSafeOstream(&std::cout, mutex.get()),
      }}) {}

/** Default ostreams are:
 * blackHole,
 * std::cerr,
 * std::cerr,
 * std::cout,
 * std::cout,
 * std::cout
 */
void Logger::setDefaultStreamsI() {
  getI(Error).ostream = &std::cerr;
  getI(Warning).ostream = &std::cerr;
  getI(Info).ostream = &std::cout;
  getI(Verbose).ostream = &std::cout;
  getI(Debug).ostream = &std::cout;
}

void Logger::setLevel(LogLevel level) { globalLogLevel = level; }

Logger::LogLevel Logger::getLevel() { return globalLogLevel; }

void Logger::setLogStream(LogLevel level, std::ostream* os) { getInstance()->setLogStreamI(level, os); }

void Logger::setLogStreamI(LogLevel level, std::ostream* os) {
  if (level == Quiet) {
    return;
  }
  std::lock_guard<std::mutex> _(*mutex);
  streams[level + 1].ostream = os;
}

ThreadSafeOstream& Logger::get(LogLevel level) {
  const auto& levelFilters = getInstance()->filters[level + 1];
  if (level <= globalLogLevel && levelFilters.empty()) {
    return getInstance()->getI(level);
  }
  return getInstance()->getI(Quiet);
}

ThreadSafeOstream& Logger::getI(LogLevel level) { return streams[level + 1]; }

Logger* Logger::getInstance() {
  static Logger instance;
  return &instance;
}

void Logger::addTagFilter(LogLevel level, const std::string& key) {
  auto& levelFilters = getInstance()->filters[level + 1];
  levelFilters.insert(key);
}

void Logger::removeTagFilter(LogLevel level, const std::string& key) {
  auto& levelFilters = getInstance()->filters[level + 1];
  levelFilters.erase(key);
}

bool Logger::isFiltered(LogLevel level, const std::string& key) {
  const auto& levelFilters = getInstance()->filters[level + 1];
  if (levelFilters.empty()) {
    return false;
  }
  return levelFilters.find(key) == levelFilters.end();
}

namespace {
void removeArgument(int& argc, char** argv, int pos) {
  argc -= 2;
  for (int i = pos; i < argc; ++i) {
    argv[i] = argv[i + 2];
  }
}
}  // namespace

void Logger::readLevelFromArgv(int& argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] == '-' && argv[i][1] == 'v' && argv[i][2] == '\0') {
      if ((i + 1 == argc) || (argv[i + 1][0] == '\0') || (argv[i + 1][1] != '\0')) {
        get(Error) << "Log level: -v takes an argument (-v <q|0|1|2|3|4>)." << std::endl;
        get(Error) << "  q:quiet; 0:error, 1:warning, 2:info, 3:verbose, 4:debug." << std::endl;
      }
      switch (argv[i + 1][0]) {
        case 'q':
          globalLogLevel = Quiet;
          removeArgument(argc, argv, i);
          return;
        case '0':
          globalLogLevel = Error;
          removeArgument(argc, argv, i);
          return;
        case '1':
          globalLogLevel = Warning;
          removeArgument(argc, argv, i);
          return;
        case '2':
          globalLogLevel = Info;
          removeArgument(argc, argv, i);
          return;
        case '3':
          globalLogLevel = Verbose;
          removeArgument(argc, argv, i);
          return;
        case '4':
          globalLogLevel = Debug;
          removeArgument(argc, argv, i);
          return;
        default:
          get(Error) << "Log level: invalid argument '" << argv[i + 1] << "' for -v. Possible values are:" << std::endl;
          get(Error) << "  q:quiet; 0:error, 1:warning, 2:info, 3:verbose, 4:debug." << std::endl;
          return;
      }
    }
  }
}

}  // namespace VideoStitch
