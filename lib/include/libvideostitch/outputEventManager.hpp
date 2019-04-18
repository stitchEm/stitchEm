// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <functional>

#include "config.hpp"

namespace VideoStitch {
namespace Output {

struct OutputEventManagerImpl;

/**
 * @brief The OutputEventManager class provides simple event system to allow plugins to communicate things to the
 * applications. User can subscribe for any event type (specified in enum) by providing void() callable, that would be
 * called when the event occurs.
 */
class VS_EXPORT OutputEventManager {
 public:
  /**
   * @brief EventType Types of events supported by OutputEventManager.
   * Note: if need be, OutputEventManager can be made generic (e.g. templatized by enum for EventTypy to accomodate
   * other set of events)
   */
  enum class EventType { Connected, Connecting, Disconnected };

  OutputEventManager();
  ~OutputEventManager();
  OutputEventManager(const OutputEventManager&) = delete;
  OutputEventManager& operator=(const OutputEventManager&) = delete;
  // Todo: default move operations after move to vs2015

  /**
   * @brief subscribe Provides a way to subscribe for event.
   * @param eventType Event type.
   * @param callback Callable that would be called when specified event occurs.
   */
  void subscribe(EventType eventType, std::function<void(const std::string&)> callback);

  /**
   * @brief publishEvent Used by plugin to publish events.
   * @param eventType Event type.
   */
  void publishEvent(EventType eventType, const std::string& payload);

  /**
   * @brief clear Unsubscribe to all the events.
   */
  void clear();

 private:
  OutputEventManagerImpl* pimpl;
};

}  // namespace Output
}  // namespace VideoStitch
