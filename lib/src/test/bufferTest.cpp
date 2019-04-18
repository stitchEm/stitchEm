// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <gpu/buffer.hpp>
#include <gpu/memcpy.hpp>
#include <gpu/uniqueBuffer.hpp>

#include "libvideostitch/gpu_device.hpp"

#include <algorithm>
#include <vector>
#include <set>

namespace VideoStitch {
namespace Testing {

std::size_t sumOfArray(const std::vector<std::size_t>& array) {
  std::size_t sum = 0;

  for (auto it : array) {
    sum += it;
  }

  return sum;
}

// Allocate host buffer through GPU API, fill with random data
// Round trip through device and compare
template <typename T>
void testHostBuffer(std::size_t bufferSize, unsigned int hostSourceFlags, unsigned int hostSinkFlninags) {
  GPU::HostBuffer<T> uninitialized;
  ENSURE(!uninitialized.hostPtr());
  GPU::HostBuffer<T> copy = uninitialized;
  ENSURE(copy == uninitialized);

  // releasing uninitialized buffer now asserts(false)
  // can no longer test for it
  // ENSURE( !copy.release() );

  auto devBuf = GPU::Buffer<T>::allocate(bufferSize, "HostBufferTest");
  ENSURE(devBuf.ok());
  ENSURE(getBufferPoolCurrentSize() == bufferSize * sizeof(T));
  ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == bufferSize * sizeof(T));

  auto potHostBuf = GPU::HostBuffer<T>::allocate(bufferSize, "HostBufferTest");
  ENSURE(potHostBuf.ok());

  ENSURE(GPU::HostBuffer<T>::getPoolSize() == bufferSize * sizeof(T));

  auto hostBuf = potHostBuf.value();
  ENSURE(hostBuf != uninitialized);

  srand(42);

  for (std::size_t i = 0; i < bufferSize; ++i) {
    hostBuf.hostPtr()[i] = static_cast<T>(rand());
  }

  auto stream = GPU::Stream::create();
  ENSURE(stream.ok());

  auto potHostBufSink = GPU::HostBuffer<T>::allocate(bufferSize, "HostBufferTest");
  ENSURE(potHostBufSink.ok());

  ENSURE(GPU::HostBuffer<T>::getPoolSize() == 2 * bufferSize * sizeof(T));

  ENSURE(GPU::memcpyAsync(devBuf.value(), hostBuf.as_const(), stream.value()));
  ENSURE(GPU::memcpyAsync(potHostBufSink.value(), devBuf.value().as_const(), stream.value()));

  stream.value().synchronize();

  ENSURE_ARRAY_EQ(hostBuf.hostPtr(), potHostBufSink.value().hostPtr(), bufferSize);

  ENSURE(stream.value().destroy());
  ENSURE(potHostBufSink.value().release());
  ENSURE(GPU::HostBuffer<T>::getPoolSize() == bufferSize * sizeof(T));
  ENSURE(hostBuf.release());
  ENSURE(GPU::HostBuffer<T>::getPoolSize() == 0);
  ENSURE(devBuf.value().release());
  ENSURE(getBufferPoolCurrentSize() == 0);
  ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == 0);
}

template <typename T>
void testHostBuffer(size_t bufferSize) {
  // test different flags only with small buffers to keep test time down
  if (bufferSize > 256) {
    testHostBuffer<T>(bufferSize, GPUHostAllocDefault, GPUHostAllocDefault);
    return;
  }

  std::set<unsigned int> flagCombo = {GPUHostAllocDefault,
                                      GPUHostAllocPinned,
                                      GPUHostAllocHostWriteOnly,
                                      GPUHostAllocDefault | GPUHostAllocPinned,
                                      GPUHostAllocDefault | GPUHostAllocHostWriteOnly,
                                      GPUHostAllocPinned | GPUHostAllocHostWriteOnly,
                                      GPUHostAllocDefault | GPUHostAllocPinned | GPUHostAllocHostWriteOnly};

  for (unsigned int sourceFlags : flagCombo) {
    for (unsigned int sinkFlags : flagCombo) {
      testHostBuffer<T>(bufferSize, sourceFlags, sinkFlags);
    }
  }
}

void testCachedBuffer(std::size_t width, std::size_t height) {
  /* XXX TODO FIXME port
  auto potCachedBuf = GPU::Cached2DBuffer<uint32_t>::allocate(width, height, "CachedBufferTest");
  ENSURE( potCachedBuf.ok() );
  ENSURE( GPU::getCachedBufferPoolCurrentSize() == width * height * sizeof(uint32_t) );
  ENSURE( sumOfArray(GPU::getCachedBufferPoolCurrentSizeByDevices()) == width * height * sizeof(uint32_t) );

  auto cachedBuf = potCachedBuf.value();
  ENSURE( cachedBuf.width() == width );
  ENSURE( cachedBuf.height() == height );

  ENSURE( cachedBuf.as_const() == cachedBuf.as_const() );

  GPU::Cached2DBuffer<uint32_t> uninitialized;
  ENSURE( !uninitialized.width() && !uninitialized.height() );
  GPU::Cached2DBuffer<uint32_t> copy = uninitialized;
  ENSURE( !copy.width() && !copy.height() );
  ENSURE( !copy.as_const().width() && !copy.as_const().height() );

  ENSURE( copy == uninitialized );
  ENSURE( copy != cachedBuf );

  // releasing uninitialized buffer now asserts(false)
  // can no longer test for it
  // ENSURE( !copy.release() );

  ENSURE( cachedBuf.release() );
  ENSURE( GPU::getCachedBufferPoolCurrentSize() == 0 );
  ENSURE( sumOfArray(GPU::getCachedBufferPoolCurrentSizeByDevices()) == 0 );
  */
}

template <typename T>
void testRoundTrip(size_t bufferSize) {
  auto buf = GPU::Buffer<T>::allocate(bufferSize, "BufferTest");
  ENSURE(buf.ok());
  ENSURE(buf.value().byteSize() == bufferSize * sizeof(T));

  srand(42);

  std::vector<T> hostData(bufferSize);
  std::generate(hostData.begin(), hostData.end(), rand);

  ENSURE(GPU::memcpyBlocking(buf.value(), hostData.data()));

  std::vector<T> fromDevice(bufferSize);

  ENSURE(GPU::memcpyBlocking(fromDevice.data(), buf.value().as_const()));

  ENSURE_ARRAY_EQ(hostData.data(), fromDevice.data(), bufferSize);

  ENSURE(buf.value().release());
}

template <typename T>
void testBufferMemset(size_t bufferSize) {
  auto buf = GPU::Buffer<T>::allocate(bufferSize, "BufferTest");
  ENSURE(buf.ok());
  ENSURE(buf.value().byteSize() == bufferSize * sizeof(T));

  srand(42);

  std::vector<T> hostData(bufferSize);
  std::generate(hostData.begin(), hostData.end(), rand);

  auto uniq = GPU::UniqueStream::create();
  ENSURE(uniq.status());
  auto stream = uniq.ref().borrow();

  ENSURE(GPU::memcpyAsync(buf.value(), hostData.data(), stream));

  ENSURE(GPU::memsetToZeroAsync(buf.value(), bufferSize * sizeof(T), stream));

  std::vector<T> fromDevice(bufferSize);
  ENSURE(GPU::memcpyAsync(fromDevice.data(), buf.value().as_const(), stream));
  stream.synchronize();

  for (size_t i = 0; i < bufferSize; i++) {
    ENSURE_EQ((T)0, fromDevice[i]);
  }

  ENSURE(buf.value().release());
}

template <typename T>
void testBuffer(size_t bufferSize) {
  auto buf = GPU::Buffer<T>::allocate(bufferSize, "BufferTest");
  ENSURE(buf.ok());
  ENSURE(buf.value().byteSize() == bufferSize * sizeof(T));
  ENSURE(buf.value().wasAllocated());
  ENSURE(getBufferPoolCurrentSize() == bufferSize * sizeof(T));
  ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == bufferSize * sizeof(T));

  GPU::Buffer<T> uninitialized;
  ENSURE(!uninitialized.wasAllocated());
  GPU::Buffer<T> copy = uninitialized;
  ENSURE(!copy.wasAllocated());
  ENSURE(!copy.as_const().wasAllocated());

  ENSURE(copy == uninitialized);
  ENSURE(copy != buf.value());

  // releasing uninitialized buffer now asserts(false)
  // can no longer test for it
  // ENSURE( !copy.release() );

  ENSURE(buf.value().release());
  ENSURE(getBufferPoolCurrentSize() == 0);
  ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == 0);
}

void testBufferCasting(size_t bufferSize) {
  auto potBuf = GPU::Buffer<uint32_t>::allocate(bufferSize, "BufferTest");
  ENSURE(potBuf.ok());

  GPU::Buffer<uint32_t> buf32 = potBuf.value();

  // Conversion from T to const T should be automatic, like with real types
  // so that a read-write Buffer can be used in read-only context
  GPU::Buffer<const uint32_t> buf32_const = buf32;
  ENSURE(buf32_const.byteSize() == buf32.byteSize());
  ENSURE(buf32_const.numElements() == buf32.numElements());
  ENSURE(buf32_const == buf32);

  // but we can make it const explicitely
  GPU::Buffer<const uint32_t> buf32_const_expl = buf32.as_const();
  ENSURE(buf32_const_expl.byteSize() == buf32.byteSize());
  ENSURE(buf32_const_expl.numElements() == buf32.numElements());
  ENSURE(buf32_const_expl == buf32);

  GPU::Buffer<uint8_t> buf8 = buf32.as<uint8_t>();
  ENSURE(buf8.byteSize() == buf32.byteSize());
  ENSURE(buf8.numElements() == buf32.numElements() * 4);
  ENSURE(buf8.as<uint32_t>() == buf32);

  GPU::Buffer<float> buf8f = buf32.as<float>();
  ENSURE(buf8f.byteSize() == buf32.byteSize());
  ENSURE(buf8f.numElements() == buf32.numElements());
  ENSURE(buf8f.as<uint32_t>() == buf32);

  ENSURE(buf32.release());
}

template <typename T>
void testUniqueBuffer(size_t bufferSize) {
  // unique buffer and its automatic GPU memory release
  {
    {
      auto potUniqBuffer = GPU::uniqueBuffer<T>(bufferSize, "UniqueBuffer test");
      ENSURE(potUniqBuffer.ok());

      auto buf = potUniqBuffer.borrow();

      ENSURE(buf.byteSize() == bufferSize * sizeof(T));
      ENSURE(buf.wasAllocated());
      ENSURE(getBufferPoolCurrentSize() == bufferSize * sizeof(T));
      ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == bufferSize * sizeof(T));
    }

    // unique buffer should have been released automatically
    ENSURE(getBufferPoolCurrentSize() == 0);
    ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == 0);
  }

  // uninitialized unique buffers (--> class members!), and borrowing them
  {
    GPU::UniqueBuffer<T> uninitialized;
    ENSURE(!uninitialized);

    {
      GPU::Buffer<T> uninitBorrow = uninitialized.borrow();
      ENSURE(!uninitBorrow.wasAllocated());
    }

    {
      auto uninitBorrowConst = uninitialized.borrow_const();
      ENSURE(!uninitBorrowConst.wasAllocated());
    }
    // make sure an uninitialized unique buffer can be destroyed without problems
  }

  // realloc
  {
    GPU::UniqueBuffer<T> uniqueBuffer;
    ENSURE(!uniqueBuffer);

    // nothing allocated yet
    ENSURE(getBufferPoolCurrentSize() == 0);
    ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == 0);

    // re-alloc of uninitialized buffer is valid
    uniqueBuffer.recreate(bufferSize, "UniqueBuffer test");

    {
      auto buf = uniqueBuffer.borrow();
      ENSURE(buf.byteSize() == bufferSize * sizeof(T));
      ENSURE(buf.wasAllocated());
      ENSURE(getBufferPoolCurrentSize() == bufferSize * sizeof(T));
      ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == bufferSize * sizeof(T));
    }

    // re-alloc of valid buffer
    auto reallocBufferSize = bufferSize + 7;
    uniqueBuffer.recreate(reallocBufferSize, "UniqueBuffer test");

    {
      auto buf = uniqueBuffer.borrow();
      ENSURE(buf.byteSize() == reallocBufferSize * sizeof(T));
      ENSURE(buf.wasAllocated());
      ENSURE(getBufferPoolCurrentSize() == reallocBufferSize * sizeof(T));
      ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == reallocBufferSize * sizeof(T));
    }

    uniqueBuffer.releaseOwnership().release();
    ENSURE(!uniqueBuffer);

    // unique buffer was manually released
    ENSURE(getBufferPoolCurrentSize() == 0);
    ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == 0);

    // allocated again
    uniqueBuffer.recreate(reallocBufferSize, "UniqueBuffer test");
    uniqueBuffer.recreate(reallocBufferSize + 3, "UniqueBuffer test");
    uniqueBuffer.recreate(reallocBufferSize + 5, "UniqueBuffer test");
    uniqueBuffer.recreate(reallocBufferSize, "UniqueBuffer test");

    {
      auto buf = uniqueBuffer.borrow();
      ENSURE(buf.byteSize() == reallocBufferSize * sizeof(T));
      ENSURE(buf.wasAllocated());
      ENSURE(getBufferPoolCurrentSize() == reallocBufferSize * sizeof(T));
      ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == reallocBufferSize * sizeof(T));
    }

    // should be released when it goes out of scope
  }

  ENSURE(getBufferPoolCurrentSize() == 0);
  ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == 0);

  // unique buffer release ownership
  {
    GPU::UniqueBuffer<T> uniqueBuffer;
    ENSURE(!uniqueBuffer);

    uniqueBuffer.alloc(bufferSize, "UniqueBuffer test");

    GPU::Buffer<T> buf = uniqueBuffer.releaseOwnership();

    ENSURE(getBufferPoolCurrentSize() == bufferSize * sizeof(T));
    ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == bufferSize * sizeof(T));

    buf.release();

    ENSURE(getBufferPoolCurrentSize() == 0);
    ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == 0);
  }

  // potential unique buffer release ownership
  {
    {
      GPU::Buffer<T> buf;

      {
        auto potUniqBuffer = GPU::uniqueBuffer<T>(bufferSize, "UniqueBuffer test");
        ENSURE(potUniqBuffer.ok());
        buf = potUniqBuffer.borrow();
        potUniqBuffer.releaseOwnership();
      }

      ENSURE(buf.byteSize() == bufferSize * sizeof(T));
      ENSURE(buf.wasAllocated());
      ENSURE(getBufferPoolCurrentSize() == bufferSize * sizeof(T));
      ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == bufferSize * sizeof(T));

      buf.release();
    }
  }

  // released manually
  ENSURE(getBufferPoolCurrentSize() == 0);
  ENSURE(sumOfArray(getBufferPoolCurrentSizeByDevices()) == 0);
}

}  // namespace Testing
}  // namespace VideoStitch

#define RUN_BUFFER_TESTS(BufferType)                              \
  VideoStitch::Testing::testBuffer<BufferType>(bufferSize);       \
  VideoStitch::Testing::testRoundTrip<BufferType>(bufferSize);    \
  VideoStitch::Testing::testHostBuffer<BufferType>(bufferSize);   \
  VideoStitch::Testing::testBufferMemset<BufferType>(bufferSize); \
  VideoStitch::Testing::testUniqueBuffer<BufferType>(bufferSize);

int main(int /*argc*/, char** /*argv*/) {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  for (std::size_t bufferSize : {1, 7, 32, 129}) {
    RUN_BUFFER_TESTS(unsigned char);
    RUN_BUFFER_TESTS(uint32_t);
    VideoStitch::Testing::testBufferCasting(bufferSize);
    std::size_t width = static_cast<std::size_t>(sqrt(static_cast<double>(bufferSize)));
    std::size_t height = static_cast<std::size_t>(pow(static_cast<double>(bufferSize), 0.45));
    VideoStitch::Testing::testCachedBuffer(width, height);
  }

  return 0;
}
