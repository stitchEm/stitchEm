// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "testing_common.hpp"

#include "util.hpp"

namespace VideoStitch {
namespace Testing {

// increase if too many random failures
const std::chrono::milliseconds timeoutInterval(5);
const std::chrono::milliseconds sleepInterval(8);

void testTimeoutHandler() {
  auto handler = std::make_unique<Util::TimeoutHandler>(timeoutInterval);

  ENSURE(Util::TimeoutHandler::checkInterrupt(handler.get()) == false, "Should not yet have timed out");

  std::this_thread::sleep_for(sleepInterval);
  ENSURE(Util::TimeoutHandler::checkInterrupt(handler.get()) == true, "Should have timed out now");

  handler->reset(timeoutInterval);
  ENSURE(Util::TimeoutHandler::checkInterrupt(handler.get()) == false, "Timeout handler was reset");

  std::this_thread::sleep_for(sleepInterval);
  ENSURE(Util::TimeoutHandler::checkInterrupt(handler.get()) == true, "Should have timed out now");

  // thread-safety
  int iterations = 10;

  handler->reset(timeoutInterval);

  std::thread resetThread([&]() {
    for (int i = 0; i < iterations; i++) {
      handler->reset(timeoutInterval);
    }
  });

  std::thread checkThread([&]() {
    for (int i = 0; i < iterations; i++) {
      ENSURE(Util::TimeoutHandler::checkInterrupt(handler.get()) == false,
             "Timeout handler is being reset on other thread");
    }
  });

  resetThread.join();
  checkThread.join();

  std::this_thread::sleep_for(sleepInterval);
  ENSURE(Util::TimeoutHandler::checkInterrupt(handler.get()) == true, "Should have timed out now");
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int /*argc*/, char** /*argv*/) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::testTimeoutHandler();
  return 0;
}
