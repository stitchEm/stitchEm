// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/config.hpp"
#include "libvideostitch/span.hpp"

namespace VideoStitch {
namespace IO {

/**
 * Data Packet.
 */
struct Packet {
  /**
   * Span to the packet data.
   */
  Span<unsigned char> data;

  /**
   * decoding timestamp in microseconds of the data packet.
   */
  mtime_t dts;
  /**
   * Presentation timestamp in microseconds of the data packet.
   */
  mtime_t pts;
};
}  // namespace IO

namespace Output {
enum class MuxerThreadStatus { OK = 0, CreateError, WriteError, EncodeError, NetworkError, TimeOutError };

}
}  // namespace VideoStitch
