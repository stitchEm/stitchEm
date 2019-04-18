// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <map>
#include <vector>
#include <mutex>

#include "libvideostitch/outputEventManager.hpp"

namespace VideoStitch {
namespace Output {

struct OutputEventManagerImpl {
  std::map<OutputEventManager::EventType, std::vector<std::function<void(const std::string&)>>> callbackMap;
  std::mutex subscribersLock;
};

void OutputEventManager::subscribe(OutputEventManager::EventType eventType,
                                   std::function<void(const std::string&)> callback) {
  std::lock_guard<std::mutex> lock(pimpl->subscribersLock);
  pimpl->callbackMap[eventType].push_back(callback);
}

void OutputEventManager::publishEvent(OutputEventManager::EventType eventType, const std::string& payload) {
  std::lock_guard<std::mutex> lock(pimpl->subscribersLock);
  for (auto& callable : pimpl->callbackMap[eventType]) {
    callable(payload);
  }
}

void OutputEventManager::clear() {
  std::lock_guard<std::mutex> lock(pimpl->subscribersLock);
  pimpl->callbackMap.clear();
}

OutputEventManager::OutputEventManager() : pimpl(new OutputEventManagerImpl()) {}

OutputEventManager::~OutputEventManager() { delete pimpl; }

}  // namespace Output
}  // namespace VideoStitch
