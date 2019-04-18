// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/gpu_device.hpp"
#include <gpu/buffer.hpp>
#include <gpu/event.hpp>
#include <gpu/memcpy.hpp>
#include <gpu/stream.hpp>

#include <algorithm>
#include <vector>

namespace VideoStitch {
namespace Testing {

void testStreamSynchronizeOnStream() {
  auto potCopyToHostStream = GPU::UniqueStream::create();
  ENSURE(potCopyToHostStream.ok());
  auto& copyToHostStream = potCopyToHostStream.ref().borrow();

  auto potGPUOperationStream = GPU::UniqueStream::create();
  ENSURE(potGPUOperationStream.ok());
  auto& opStream = potGPUOperationStream.ref().borrow();

  srand(42);

  size_t bufferSize = 3511;

  std::vector<uint32_t> hostData(bufferSize);
  std::generate(hostData.begin(), hostData.end(), rand);

  auto srcBuf = GPU::Buffer<uint32_t>::allocate(bufferSize, "BufferTest");
  ENSURE(srcBuf.ok());

  ENSURE(GPU::memcpyBlocking(srcBuf.value(), hostData.data()));

  std::vector<uint32_t> hostDataDst(bufferSize);
  std::generate(hostDataDst.begin(), hostDataDst.end(), rand);

  auto dstBuf = GPU::Buffer<uint32_t>::allocate(bufferSize, "BufferTest");
  ENSURE(dstBuf.ok());

  // srcBuf filled with hostData, let's start an async copy on opStream device src to device dst
  ENSURE(GPU::memcpyAsync(dstBuf.value(), srcBuf.value().as_const(), srcBuf.value().byteSize(), opStream));

  // copyToHostStream has to wait for opStream to finish current operation
  ENSURE(copyToHostStream.synchronizeOnStream(opStream));
  ENSURE(GPU::memcpyAsync(hostDataDst.data(), dstBuf.value().as_const(), srcBuf.value().byteSize(), copyToHostStream));

  ENSURE(copyToHostStream.synchronize());

  // if synchronizeOnStream worked as expected, the dev->host copy waited for the dev->dev copy
  // if it didn't wait the data is probably 0
  ENSURE_ARRAY_EQ(hostData.data(), hostDataDst.data(), (int)bufferSize);

  ENSURE(srcBuf.value().release());
  ENSURE(dstBuf.value().release());
}

void testCPUWaitForEvent() {
  size_t bufferSize = 1021;
  auto buf = GPU::Buffer<uint32_t>::allocate(bufferSize, "BufferTest");
  ENSURE(buf.ok());

  auto potStream = GPU::UniqueStream::create();
  ENSURE(potStream.ok());
  auto stream = potStream.ref().borrow();

  srand(42);

  std::vector<uint32_t> hostData(bufferSize);
  std::generate(hostData.begin(), hostData.end(), rand);

  ENSURE(GPU::memcpyAsync(buf.value(), hostData.data(), stream));

  std::vector<uint32_t> fromDevice(bufferSize);

  ENSURE(GPU::memcpyAsync(fromDevice.data(), buf.value().as_const(), stream));

  auto event = stream.recordEvent();
  ENSURE(event.status());

  ENSURE(event.value().synchronize());

  ENSURE_ARRAY_EQ(hostData.data(), fromDevice.data(), bufferSize);

  ENSURE(buf.value().release());
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int /*argc*/, char** /*argv*/) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  VideoStitch::Testing::testStreamSynchronizeOnStream();

  VideoStitch::Testing::testCPUWaitForEvent();
  return 0;
}
