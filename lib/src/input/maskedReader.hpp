// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "statefulReader.hpp"

namespace VideoStitch {
namespace Input {
/**
 * A delegator reader that masks the input.
 */
class MaskedReader : public StatefulReader<GPU::Buffer<unsigned char>> {
 public:
  static MaskedReader* create(VideoReader* delegate, const unsigned char* maskHostBuffer);
  ~MaskedReader();

  ReadStatus readFrame(mtime_t& date, unsigned char* video) { return delegate->readFrame(date, video); }
  Status seekFrame(frameid_t frame) { return delegate->seekFrame(frame); }
  Status unpackDevBuffer(GPU::Surface& dst, const GPU::Buffer<const unsigned char>& src, GPU::Stream& stream) const;

  const VideoReader* getDelegate() const { return delegate; }
  Status perThreadInit();
  void perThreadCleanup();

 private:
  MaskedReader(VideoReader* delegate, unsigned char* maskHostBuffer);

  VideoReader* const delegate;
  unsigned char* const maskHostBuffer;  // owned
};
}  // namespace Input
}  // namespace VideoStitch
