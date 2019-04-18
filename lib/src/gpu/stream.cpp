// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/stream.hpp"

namespace VideoStitch {
namespace GPU {

Status Stream::synchronizeOnStream(const Stream& other) const {
  PotentialValue<Event> syncEvent = other.recordEvent();
  FAIL_RETURN(syncEvent.status());
  return waitOnEvent(syncEvent.value());
}

PotentialValue<UniqueStream> UniqueStream::create() {
  PotentialValue<Stream> potStream = Stream::create();
  FAIL_RETURN(potStream.status());
  return PotentialValue<UniqueStream>(UniqueStream(potStream.value()));
}

}  // namespace GPU
}  // namespace VideoStitch
