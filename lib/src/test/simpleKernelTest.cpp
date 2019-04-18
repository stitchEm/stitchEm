// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Basic input unpacking tests.

#include "gpu/testing.hpp"
#include "gpu/exampleKernel.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/uniqueBuffer.hpp"
#include "libvideostitch/context.hpp"
#include "util/pnm.hpp"

#include <cmath>
#include <future>
#include <thread>

namespace VideoStitch {
namespace Testing {

static const int NUM_PARALLEL_INVOCATIONS = 3;
static const int NUM_ITERATIONS = 3;

void testDummyKernel(float mult) {
  const int nbElements = 123;

  // init input
  float mem_host[nbElements];
  for (int i = 0; i < nbElements; ++i) {
    mem_host[i] = static_cast<float>(i);
  }
  auto inputBuff = GPU::uniqueBuffer<float>(nbElements, "dummyKernelTest");
  ENSURE(inputBuff.status());

  auto outputBuff = GPU::uniqueBuffer<float>(nbElements, "dummyKernelTest");
  ENSURE(outputBuff.status());

  auto potUniqStream = GPU::UniqueStream::create();
  ENSURE(potUniqStream.status());
  auto stream = potUniqStream.ref().borrow();

  // transfer and compute
  ENSURE(GPU::memcpyAsync(inputBuff.borrow(), mem_host, stream));
  Status status = Core::callDummyKernel(outputBuff.borrow(), inputBuff.borrow(), nbElements, mult, stream);
  ENSURE(status);
  ENSURE(GPU::memcpyAsync(mem_host, outputBuff.borrow().as_const(), stream));
  stream.synchronize();

  // compare results
  for (int i = 0; i < nbElements; ++i) {
    float targetValue = mult * static_cast<float>(i);
    ENSURE_APPROX_EQ(mem_host[i], targetValue, 1e-6f);
  }
}

void testParallelKernelCall() {
  std::vector<std::thread> threads;
  ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));
  for (int start = 0; start <= NUM_PARALLEL_INVOCATIONS; start++) {
    std::thread t([=]() {
      ENSURE(VideoStitch::GPU::useDefaultBackendDevice());
      for (int it = 0; it <= NUM_ITERATIONS; it++) {
        testDummyKernel(start * it);
      }
    });
    threads.push_back(std::move(t));
  }
  for (std::thread& t : threads) {
    t.join();
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));
  VideoStitch::Testing::testDummyKernel(2.f);

#ifndef OCLGRIND
  // current oclgrind implementation does not seem to handle parallel kernel invocations correctly
  VideoStitch::Testing::testParallelKernelCall();
#endif
  return 0;
}
