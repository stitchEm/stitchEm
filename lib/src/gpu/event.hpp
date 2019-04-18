// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/status.hpp"

namespace VideoStitch {
namespace GPU {

/** Events inform observers that a GPU Stream has reached a point of computation.
 *  Event objects can be created by calling Stream::record().
 *
 *  Block host: the CPU can wait for an Event to be triggered by calling .synchronize()
 *
 *  Block GPU: Streams can be configured to wait for Events to trigger before continuing
 *  their computations (see Stream API waitOnEvent).
 */
class Event {
 public:
  /** Let the current CPU thread wait until the GPU Stream
   *  has reached the point where the Event was recorded.
   */
  Status synchronize();

  ~Event();

  Event(const Event& other);

  void swap(Event& other) { std::swap(pimpl, other.pimpl); }

  Event operator=(Event other) {
    swap(other);
    return *this;
  }

  /* Default constructed, empty wrapper.
   * Not a valid GPU Event. Can not be used to do GPU synchronization.
   * Use Stream::recordEvent() to create an Event.
   */
  Event();

  class DeviceEvent;

 private:
  DeviceEvent* pimpl;
  explicit Event(DeviceEvent* pimpl);

 public:
  /** Provide the GPU backend implementation with simple access to the underlying data structure. */
  const DeviceEvent& get() const;
};

}  // namespace GPU
}  // namespace VideoStitch
