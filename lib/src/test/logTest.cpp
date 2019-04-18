// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <gpu/buffer.hpp>
#include "libvideostitch/logging.hpp"
#include <sstream>

namespace VideoStitch {
namespace Testing {

void testLog() {
  Logger::get(Logger::Error) << "log error" << std::endl;
  Logger::get(Logger::Warning) << "log warning" << std::endl;
  Logger::get(Logger::Info) << "log info" << std::endl;
  Logger::get(Logger::Verbose) << "log verbose" << std::endl;
  Logger::get(Logger::Debug) << "log debug" << std::endl;

  // Stream change
  {
    std::ostringstream str;
    std::string test = "12345\n";
    Logger::setLogStream(Logger::Info, &str);
    Logger::get(Logger::Info) << test;
    ENSURE_EQ(str.str(), test);
  }

  // Output level
  {
    std::ostringstream loggingErrorStr;
    std::ostringstream loggingWarningStr;
    std::ostringstream loggingInfoStr;
    std::ostringstream loggingVerboseStr;
    std::ostringstream loggingDebugStr;
    Logger::setLogStream(Logger::Error, &loggingErrorStr);
    Logger::setLogStream(Logger::Warning, &loggingWarningStr);
    Logger::setLogStream(Logger::Info, &loggingInfoStr);
    Logger::setLogStream(Logger::Verbose, &loggingVerboseStr);
    Logger::setLogStream(Logger::Debug, &loggingDebugStr);

    int i = 0;
    Logger::get(Logger::Quiet) << "String " << i++ << std::endl;
    ENSURE_EQ(loggingErrorStr.str(), std::string());
    ENSURE_EQ(loggingWarningStr.str(), std::string());
    ENSURE_EQ(loggingInfoStr.str(), std::string());
    ENSURE_EQ(loggingVerboseStr.str(), std::string());
    ENSURE_EQ(loggingDebugStr.str(), std::string());

    Logger::get(Logger::Warning) << "String " << i++ << std::endl;
    ENSURE_EQ(loggingErrorStr.str(), std::string());
    ENSURE_EQ(loggingWarningStr.str(), std::string("String 1\n"));
    ENSURE_EQ(loggingInfoStr.str(), std::string());
    ENSURE_EQ(loggingVerboseStr.str(), std::string());
    ENSURE_EQ(loggingDebugStr.str(), std::string());

    Logger::get(Logger::Error) << "String " << i++ << std::endl;
    ENSURE_EQ(loggingErrorStr.str(), std::string("String 2\n"));
    ENSURE_EQ(loggingWarningStr.str(), std::string("String 1\n"));
    ENSURE_EQ(loggingInfoStr.str(), std::string());
    ENSURE_EQ(loggingVerboseStr.str(), std::string());
    ENSURE_EQ(loggingDebugStr.str(), std::string());

    loggingErrorStr.str(std::string());
    Logger::error("TAG") << "String " << i++ << std::endl;
    ENSURE_EQ(loggingErrorStr.str(), std::string(Logger::concatenateTags("TAG") + "String 3\n"));
    ENSURE_EQ(loggingWarningStr.str(), std::string("String 1\n"));
    ENSURE_EQ(loggingInfoStr.str(), std::string());
    ENSURE_EQ(loggingVerboseStr.str(), std::string());
    ENSURE_EQ(loggingDebugStr.str(), std::string());
  }

  // Filters

  {
    std::ostringstream str;
    Logger::setLogStream(Logger::Warning, &str);

    Logger::addTagFilter(Logger::Warning, "AAA");
    Logger::get(Logger::Warning) << "Hello\n";
    ENSURE_EQ(str.str(), std::string());

    Logger::get(Logger::Warning, "BBB") << "Hello\n";
    ENSURE_EQ(str.str(), std::string());

    Logger::get(Logger::Warning, "AAA") << "Hello\n";
    ENSURE_EQ(str.str(), std::string("[AAA] Hello\n"));

    str.str(std::string());
    Logger::addTagFilter(Logger::Warning, "BBB");
    Logger::removeTagFilter(Logger::Warning, "AAA");
    Logger::get(Logger::Warning, "AAA") << "Hello\n";
    ENSURE_EQ(str.str(), std::string());

    Logger::removeTagFilter(Logger::Warning, "BBB");
    Logger::get(Logger::Warning, "AAA") << "Hello\n";
    ENSURE_EQ(str.str(), std::string("[AAA] Hello\n"));
  }
}
}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char** argv) {
  VideoStitch::Testing::initTest();
  VideoStitch::Logger::readLevelFromArgv(argc, argv);
  VideoStitch::Testing::testLog();
  return 0;
}
