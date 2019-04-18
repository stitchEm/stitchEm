// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "event.hpp"

#include "libvideostitch/status.hpp"

#ifndef VS_OPENCL
#include <cuda_runtime.h>
#endif

#include <memory>

namespace VideoStitch {
namespace GPU {

/** A wrapper around a stream or command queue on the GPU.
 *  For a set of dependent computations, create a stream,
 *  enqueue copy commands and kernel calls on it, read back
 *  the data, then destroy the stream.
 *
 *  The Stream wrapper object does not own or influence the lifetime
 *  of the underlying GPU stream/command queue.
 *  The lifetime is determined by the create/destroy functions.
 *
 *  If you need a Stream that owns the underlying GPU stream
 *  and destroys it when it goes out of scope, use UniqueStream.
 */
class Stream {
 public:
  /* Default constructed, empty wrapper.
   * Not a valid GPU Stream. Can not be used to do GPU operations.
   * Use Stream::create() to create a valid Stream.
   */
  Stream();

  ~Stream() {}

  Stream(const Stream& other) = default;

// this is here for compatibility with code that has not been ported yet
// but uses functionality that is already ported to GPU API
// TODO_OPENCL_IMPL: remove this constructor after porting is finished
// TODO leaks pimpl, as Stream pimpl lifetime is that
// must use destroyDeprecatedCUDAWrapper after usage
#ifndef VS_OPENCL
  explicit Stream(cudaStream_t cudaStream);
  void destroyDeprecatedCUDAWrapper();
#endif

  /** A Stream for common GPU operations, e.g. debugging.
   *  Valid throughout the program run. Must not be destroyed.
   */
  static Stream getDefault();

  /** Creates a new Stream for GPU operations.
   *  Use destroy() to release it after usage.
   */
  static PotentialValue<Stream> create();

  /** Destroy the underlying GPU stream.
   *  All GPU::Stream wrappers referencing the underlying GPU stream
   *  will become invalid.
   */
  Status destroy();

  /** Issues all previously queued GPU commands in a stream to
   *  the device associated with the stream.
   */
  Status flush() const;

  /** Block the current CPU thread until all currently enqueued operations
   *  on the GPU on this Stream have finished.
   */
  Status synchronize() const;

  /** Ensure that everything running on other Stream is finished
   *  before continuing with operations in this Stream.
   *  This does not block the CPU.
   */
  Status synchronizeOnStream(const Stream& other) const;

  /** Enqueue a barrier on the Stream that will cause the stream to pause
   *  until the event is triggered.
   */
  Status waitOnEvent(Event event) const;

  /** Create an Event that will fire once the GPU
   *  has finished with all operations currently scheduled
   *  on this Stream
   */
  PotentialValue<Event> recordEvent() const;

  bool operator==(const Stream& other) const;

  bool operator!=(const Stream& other) const { return !(*this == other); }

  friend class UniqueStream;

 private:
  class DeviceStream;
  DeviceStream* pimpl;

 public:
  /** Provide the GPU backend implementation with simple access to the underlying data structure. */
  const DeviceStream& get() const;
};

/** A Stream that automatically calls destroy() on itself when
    the wrapper goes out of scope / is deleted.
 */
class UniqueStream {
 public:
  UniqueStream() {}

  /** Take ownership of Stream s */
  explicit UniqueStream(Stream s) : stream(s) {}

  UniqueStream(UniqueStream&& other) : stream(other.stream) { other.stream = Stream(); }

  /** Create a UniqueStream that destroys underlying GPU
   *  data structure when going out of scope.
   */
  static PotentialValue<UniqueStream> create();

  /** Access to GPU::Stream, does not prolong lifetime. */
  const Stream& borrow() const { return stream; }

  ~UniqueStream() {
    if (stream.pimpl) {
      stream.destroy();
      stream = Stream();
    }
  }

 private:
  Stream stream;
};

}  // namespace GPU
}  // namespace VideoStitch
