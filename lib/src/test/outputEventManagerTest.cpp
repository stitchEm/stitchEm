// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "libvideostitch/outputEventManager.hpp"

namespace VideoStitch {
namespace Testing {
void testSubscribePublish() {
  Output::OutputEventManager manager;
  std::string result_payload;
  manager.subscribe(Output::OutputEventManager::EventType::Connecting,
                    [&result_payload](std::string payload) { result_payload = payload; });
  manager.publishEvent(Output::OutputEventManager::EventType::Connecting, "wheee");

  ENSURE_EQ(result_payload, std::string("wheee"));
}

void testEmptyPublish() {
  // should do nothing but should not crush
  Output::OutputEventManager manager;
  manager.publishEvent(Output::OutputEventManager::EventType::Connecting, "wheee");
  ENSURE(true);
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testEmptyPublish();
  VideoStitch::Testing::testSubscribePublish();

  return 0;
}
