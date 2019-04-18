// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/gpu_device.hpp"
#include <gpu/stream.hpp>

namespace VideoStitch {
namespace Testing {

void testStreamLifetime() {
  auto stream = GPU::Stream::create();
  ENSURE(stream.ok());
  ENSURE(stream.value().destroy());
}

void testDefaultStream() {
  auto stream = GPU::Stream::create();
  ENSURE(stream.ok());
  ENSURE(GPU::Stream::getDefault() != stream.value());
  ENSURE(stream.value().destroy());
  ENSURE(GPU::Stream::getDefault() == GPU::Stream::getDefault());
}

void borrowUniqueStream(GPU::Stream stream) { ENSURE(stream.synchronize()); }

void testUniqueStream() {
  auto potUniqueStream = GPU::UniqueStream::create();
  ENSURE(potUniqueStream.ok());

  auto& uniqueStream = potUniqueStream.ref();

  // Stream we created should not be default stream
  ENSURE(GPU::Stream::getDefault() != uniqueStream.borrow());

  // borrowing twice should borrow same underlying stream
  ENSURE(potUniqueStream.ref().borrow() == potUniqueStream.ref().borrow());

  borrowUniqueStream(uniqueStream.borrow());
}

void testUninitializedStream() {
  GPU::Stream uninitialized;
  GPU::Stream copy = uninitialized;
  ENSURE(copy == uninitialized);
  ENSURE(copy != GPU::Stream::getDefault());
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int /*argc*/, char** /*argv*/) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  VideoStitch::Testing::testStreamLifetime();
  VideoStitch::Testing::testDefaultStream();
  VideoStitch::Testing::testUniqueStream();
  VideoStitch::Testing::testUninitializedStream();
  return 0;
}
