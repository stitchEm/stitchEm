// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "maskedReader.hpp"

#include <gpu/buffer.hpp>
#include <gpu/memcpy.hpp>

#include "gpu/input/maskInput.hpp"

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>

namespace VideoStitch {
namespace Input {

MaskedReader* MaskedReader::create(VideoReader* delegate, const unsigned char* maskHostBuffer) {
  if (!maskHostBuffer) {
    return NULL;
  }
  const size_t bufferSize = (size_t)(delegate->getSpec().width * delegate->getSpec().height);
  unsigned char* ownedBuffer = new unsigned char[bufferSize];
  if (!ownedBuffer) {
    return NULL;
  }
  memcpy(ownedBuffer, maskHostBuffer, bufferSize);
  return new MaskedReader(delegate, ownedBuffer);
}

MaskedReader::MaskedReader(VideoReader* delegate, unsigned char* maskHostBuffer)
    : Reader(delegate->id),
      StatefulBase(delegate->getSpec().width, delegate->getSpec().height, delegate->getFrameDataSize(),
                   delegate->getSpec().format, delegate->getSpec().addressSpace, delegate->getSpec().frameRate,
                   delegate->getFirstFrame(), delegate->getLastFrame(), delegate->getSpec().frameRateIsProcedural,
                   maskHostBuffer),
      delegate(delegate),
      maskHostBuffer(maskHostBuffer) {}

MaskedReader::~MaskedReader() {
  delete[] maskHostBuffer;
  delete delegate;
}

Status MaskedReader::perThreadInit() {
  FAIL_RETURN(delegate->perThreadInit());
  const size_t bufferSize = (size_t)(getSpec().width * getSpec().height);
  auto maskDevBuffer = GPU::Buffer<unsigned char>::allocate(bufferSize, "Masked Reader");
  if (!maskDevBuffer.ok()) {
    return maskDevBuffer.status();
  }
  FAIL_RETURN(GPU::memcpyBlocking(maskDevBuffer.value(), maskHostBuffer, bufferSize));
  return setCurrentDeviceData(maskDevBuffer.value());
}

void MaskedReader::perThreadCleanup() {
  delegate->perThreadCleanup();
  auto maskDevBufferP = getCurrentDeviceData();
  if (maskDevBufferP) {
    maskDevBufferP->release();
  }
  StatefulBase::perThreadCleanup();
}

Status MaskedReader::unpackDevBuffer(GPU::Surface& dst, const GPU::Buffer<const unsigned char>& src,
                                     GPU::Stream& stream) const {
  FAIL_RETURN(delegate->unpackDevBuffer(dst, src, stream));
  auto maskDevBufferP = getCurrentDeviceData();
  assert(maskDevBufferP);
  if (!maskDevBufferP) {
    // We should always have gone through perThreadInit().
    return {Origin::Input, ErrType::ImplementationError, "Uninitialized masked reader"};
  }

  maskInput(dst, *maskDevBufferP, (unsigned)getSpec().width, (unsigned)getSpec().height, stream);
  return Status::OK();
}
}  // namespace Input
}  // namespace VideoStitch

/// @endcond
