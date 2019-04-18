// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <core/bufferedReader.hpp>

/** A test for the BufferedReader implementation.
 *
 * The MockReader increases its date with each readFrame call
 * and sets the first Byte of audio and video data to the current date.
 * These values can be compared to ensure the frames integrity after it
 * is passed through the BufferedReader, and that the frames remain in order.
 * Note: it's a Byte value, so there's overflow over frame 255
 *
 * Further than the basic operations, there are tests to make sure there are no deadlocks
 * or race conditions when using the BufferedReader from different threads.
 */

namespace VideoStitch {
namespace Testing {

static int64_t readerWidthMock = 512;
static int64_t readerHeightMock = 256;

static int64_t frameDataSizeMock = readerWidthMock * readerHeightMock * 4;

static FrameRate frameRateMock = {25, 1};

static frameid_t firstFrameMock = 0;
static frameid_t lastFrameMock = 1000;

static const unsigned char* maskHostBufferMock = nullptr;

// how much time to sleep per time-sensitive testing step?
// increase if test fails due to being too short
static const std::chrono::microseconds tickLength{1000};

class MockReader : public Input::VideoReader {
 public:
  explicit MockReader(std::chrono::microseconds readFrameTime)
      : Reader(0),
        VideoReader(readerWidthMock, readerHeightMock, frameDataSizeMock, PixelFormat::RGBA, Host, frameRateMock,
                    firstFrameMock, lastFrameMock, true /* procedural */, maskHostBufferMock),
        readFrameTime(readFrameTime) {}

  MockReader() : MockReader(std::chrono::microseconds(0)) {}

  virtual ~MockReader() {}

  virtual Input::ReadStatus readFrame(mtime_t& date, unsigned char* dst) {
    std::this_thread::sleep_for(readFrameTime);
    date = currentFrameID;
    *dst = (unsigned char)currentFrameID;

    currentFrameID++;
    return Input::ReadStatus::OK();
  }

  virtual Status seekFrame(frameid_t targetFrame) {
    std::this_thread::sleep_for(5 * readFrameTime);
    currentFrameID = targetFrame;
    return Status::OK();
  }

 private:
  int currentFrameID = 0;
  std::chrono::microseconds readFrameTime;
};

class StatusReader : public Input::VideoReader {
 public:
  StatusReader()
      : Reader(0),
        VideoReader(readerWidthMock, readerHeightMock, frameDataSizeMock, PixelFormat::RGBA, Host, frameRateMock,
                    firstFrameMock, lastFrameMock, true /* procedural */, maskHostBufferMock) {}
  virtual ~StatusReader() {}

  virtual Input::ReadStatus readFrame(mtime_t& date, unsigned char* /* dst */) {
    date = readFrameStatus;
    return Status{Origin::Input, ErrType::RuntimeError, std::to_string(readFrameStatus++)};
  }

  virtual Status seekFrame(frameid_t) {
    readFrameStatus = 0;
    return {Origin::Input, ErrType::RuntimeError, std::to_string(seekFrameStatus++)};
  }

 private:
  int readFrameStatus = 0;
  int seekFrameStatus = 0;
};

void ensureFrameIntegrity(Core::InputFrame& frame) {
  ENSURE(frame.readerStatus.ok(), "frame should have Status OK");
  ENSURE_EQ((unsigned char)frame.date, (unsigned char)*frame.buffer.hostBuffer().hostPtr(),
            "frame content should be equal to reported date");
  ENSURE_EQ(frame.buffer.hostBuffer().byteSize(), (size_t)frameDataSizeMock,
            "size of loaded frame should be equal frameDataSize");
}

void ensureFrameIntegrity(Core::InputFrame& frame, mtime_t expectedDate) {
  ensureFrameIntegrity(frame);
  ENSURE(frame.date == expectedDate, "unexpected frame time stamp");
}

// test creating a buffered reader
// test general loading / reloading behavior
void testBufferedReaderSetup() {
  std::shared_ptr<MockReader>(testReader) = std::make_shared<MockReader>();

  int numCachedFrames = 1;
  auto potBufReader = Core::BufferedReader::create(testReader, numCachedFrames);
  ENSURE(potBufReader.status(), "BufferedReader should have been created correctly");

  auto bufReader = potBufReader.object();

  ENSURE(bufReader->getDelegate() == testReader, "BufferedReader delegate getter");

  auto frame = bufReader->load();

  ensureFrameIntegrity(frame, 0);

  auto reloadFrame = bufReader->reload();
  ENSURE(frame == reloadFrame, "reloaded frame should be equal last loaded frame");
  ensureFrameIntegrity(reloadFrame, 0);
  bufReader->releaseBuffer(reloadFrame.buffer);

  bufReader->releaseBuffer(frame.buffer);

  frame = bufReader->load();
  ensureFrameIntegrity(frame, 1);

  reloadFrame = bufReader->reload();
  ENSURE(frame == reloadFrame, "reloaded frame should be equal last loaded frame");
  ensureFrameIntegrity(reloadFrame, 1);
  bufReader->releaseBuffer(reloadFrame.buffer);

  bufReader->releaseBuffer(frame.buffer);
}

// test that `numCachedFrames` are buffered and available
// without first releasing previously used frames
// check time stamp on buffered frames
void testReaderBuffering(int numCachedFrames) {
  {
    auto potBufReader = Core::BufferedReader::create(std::make_shared<MockReader>(), numCachedFrames);
    ENSURE(potBufReader.status(), "BufferedReader should have been created correctly");
    auto bufReader = potBufReader.object();

    auto frame = bufReader->load();

    ensureFrameIntegrity(frame, 0);

    auto reloadFrame = bufReader->reload();
    ENSURE(frame == reloadFrame, "reloaded frame should be equal last loaded frame");
    ensureFrameIntegrity(reloadFrame, 0);
    bufReader->releaseBuffer(reloadFrame.buffer);

    std::vector<Core::InputFrame> framesToRelease;
    framesToRelease.push_back(std::move(frame));

    for (int i = 0; i < numCachedFrames; i++) {
      frame = bufReader->load();
      ensureFrameIntegrity(frame, i + 1);

      for (int h = 0; h < 5; h++) {
        reloadFrame = bufReader->reload();
        ensureFrameIntegrity(frame, i + 1);
        ensureFrameIntegrity(reloadFrame, i + 1);
        ENSURE(frame == reloadFrame, "reloaded frame should be equal last loaded frame");
        bufReader->releaseBuffer(reloadFrame.buffer);
      }

      // check past frames are not overwritten
      for (auto loadedFrame : framesToRelease) {
        ensureFrameIntegrity(loadedFrame);
      }

      framesToRelease.push_back(std::move(frame));
    }

    for (size_t i = 0; i + 1 < framesToRelease.size(); i++) {
      bufReader->releaseBuffer(framesToRelease[i].buffer);
    }

    bufReader->releaseBuffer(framesToRelease.back().buffer);
  }
}

// test reloading frame after it has been marked for release
void testBufferedReaderReload() {
  int numCachedFrames = 1;
  auto potBufReader = Core::BufferedReader::create(std::make_shared<MockReader>(tickLength), numCachedFrames);
  ENSURE(potBufReader.status(), "BufferedReader should have been created correctly");
  auto bufReader = potBufReader.object();

  // nothing to reload yet, make sure we don't crash
  auto reloadedEmpty = bufReader->reload();
  ENSURE(!reloadedEmpty.readerStatus.ok(), "reloading before loading a frame should not be Status::OK");
  ENSURE(reloadedEmpty.buffer.rawPtr() == nullptr, "reloading before loading a frame should return nullptr frame");
  bufReader->releaseBuffer(reloadedEmpty.buffer);

  auto frame = bufReader->load();
  ensureFrameIntegrity(frame, 0);

  bufReader->releaseBuffer(frame.buffer);

  std::this_thread::sleep_for(tickLength * 2);

  auto reloadFrame = bufReader->reload();
  ENSURE(frame == reloadFrame, "reloaded frame should be equal last loaded frame");
  ensureFrameIntegrity(frame, 0);
  ensureFrameIntegrity(reloadFrame, 0);
  bufReader->releaseBuffer(reloadFrame.buffer);
}

// return frame in different thread when depleted, make sure it's not blocking further loading
void testReturnOnThread() {
  int numCachedFrames = 1;

  auto potBufReader = Core::BufferedReader::create(std::make_shared<MockReader>(tickLength), numCachedFrames);
  ENSURE(potBufReader.status(), "BufferedReader should have been created correctly");
  auto bufReader = potBufReader.object();

  auto frame0 = bufReader->load();
  auto frame1 = bufReader->load();

  std::thread releaseBufferThread([&]() {
    std::this_thread::sleep_for(tickLength * 2);
    bufReader->releaseBuffer(frame0.buffer);
  });

  // blocks because there's no buffer available
  // should eventually unblock when releaseBufferThread runs
  auto frame2 = bufReader->load();

  bufReader->releaseBuffer(frame1.buffer);
  bufReader->releaseBuffer(frame2.buffer);

  releaseBufferThread.join();
}

// test frame date after seeking
void testLoadAfterSeek() {
  int numCachedFrames = 1;
  auto potBufReader = Core::BufferedReader::create(std::make_shared<MockReader>(tickLength), numCachedFrames);
  ENSURE(potBufReader.status(), "BufferedReader should have been created correctly");
  auto bufReader = potBufReader.object();

  std::this_thread::sleep_for(tickLength * 3);

  auto seekStatus = bufReader->seekFrame(100);

  auto frame100 = bufReader->load();
  ensureFrameIntegrity(frame100, 100);

  auto frame101 = bufReader->load();
  ensureFrameIntegrity(frame101, 101);

  bufReader->releaseBuffer(frame100.buffer);
  bufReader->releaseBuffer(frame101.buffer);
}

// test seeking when no frames are available
void testLoadAfterSeekBlocking() {
  const std::chrono::microseconds readFrameLength{10000};

  int numCachedFrames = 1;

  auto potBufReader = Core::BufferedReader::create(std::make_shared<MockReader>(readFrameLength), numCachedFrames);
  ENSURE(potBufReader.status(), "BufferedReader should have been created correctly");
  auto bufReader = potBufReader.object();

  auto frame0 = bufReader->load();
  auto frame1 = bufReader->load();
  auto frame2 = bufReader->load();

  // currently blocked until new frame becomes available
  std::thread loadThread([&]() {
    auto frame100 = bufReader->load();
    ensureFrameIntegrity(frame100, 100);
    bufReader->releaseBuffer(frame100.buffer);
  });

  auto seekStatus = bufReader->seekFrame(100);
  bufReader->releaseBuffer(frame0.buffer);
  bufReader->releaseBuffer(frame1.buffer);
  bufReader->releaseBuffer(frame2.buffer);

  // give loadThread chance to load first
  std::this_thread::sleep_for(readFrameLength * 3);

  auto frame101 = bufReader->load();
  ensureFrameIntegrity(frame101, 101);
  bufReader->releaseBuffer(frame101.buffer);

  loadThread.join();
}

// test thread safety
void testRaceConditions(int numCachedFrames) {
  // really quick operations
  std::chrono::microseconds readFrameTime{10};

  auto potBufReader = Core::BufferedReader::create(std::make_shared<MockReader>(readFrameTime), numCachedFrames);
  ENSURE(potBufReader.status(), "BufferedReader should have been created correctly");
  auto bufReader = potBufReader.object();

  int iterations = 5;

  std::mutex m;
  bool shouldStart = false;
  std::condition_variable cv;
  auto pred = [&]() { return shouldStart; };

  std::atomic<int> started{0};

  std::thread loadingThread_A([&]() {
    {
      std::unique_lock<std::mutex> lock(m);
      cv.wait(lock, pred);
    }
    started++;
    for (int i = 0; i < iterations; i++) {
      auto frame = bufReader->load();
      ensureFrameIntegrity(frame);
      bufReader->releaseBuffer(frame.buffer);
    }
  });

  std::thread loadingThread_B([&]() {
    {
      std::unique_lock<std::mutex> lock(m);
      cv.wait(lock, pred);
    }
    started++;
    for (int i = 0; i < iterations; i++) {
      auto frame = bufReader->load();
      ensureFrameIntegrity(frame);
      bufReader->releaseBuffer(frame.buffer);
    }
  });

  std::thread reloadThread([&]() {
    // make sure there's at least one frame ready for reloading
    auto frame = bufReader->load();
    bufReader->releaseBuffer(frame.buffer);

    {
      std::unique_lock<std::mutex> lock(m);
      cv.wait(lock, pred);
    }
    started++;

    for (int i = 0; i < iterations; i++) {
      frame = bufReader->reload();
      ensureFrameIntegrity(frame);
      bufReader->releaseBuffer(frame.buffer);
    }
  });

  std::thread seekThread([&]() {
    {
      std::unique_lock<std::mutex> lock(m);
      cv.wait(lock, pred);
    }
    started++;

    srand(42);
    for (int i = 0; i < iterations; i++) {
      // careful not to go to close to numerical_limit<unsigned char> or the frame date check fails
      int frameToSeekTo = rand() % 100;
      ENSURE(bufReader->seekFrame(frameToSeekTo), "seeking should be successful");
      auto frame = bufReader->load();
      ensureFrameIntegrity(frame);
      ENSURE(frame.date >= frameToSeekTo, "should have sought to the correct date");
      bufReader->releaseBuffer(frame.buffer);
    }
  });

  {
    std::lock_guard<std::mutex> lock(m);
    shouldStart = true;
  }

  while (started < 4) {
    cv.notify_all();
  }

  loadingThread_A.join();
  loadingThread_B.join();
  reloadThread.join();
  seekThread.join();
}

// destroy buffered reader while seeking
void testDestroyWhileSeeking() {
  auto readFrameTime = 2 * tickLength;

  {
    int numCachedFrames = 1;

    auto potBufReader = Core::BufferedReader::create(std::make_shared<MockReader>(readFrameTime), numCachedFrames);
    ENSURE(potBufReader.status(), "BufferedReader should have been created correctly");
    auto bufReader = potBufReader.object();

    auto frame0 = bufReader->load();
    bufReader->releaseBuffer(frame0.buffer);

    std::thread seekThread([&]() { ENSURE(bufReader->seekFrame(23), "seeking should be successful"); });
    seekThread.detach();

    std::this_thread::sleep_for(tickLength);
    // destroy buffered reader
  }

  std::this_thread::sleep_for(5 * readFrameTime + tickLength);
}

// destroy buffered reader while reading
void testDestroyWhileReading() {
  auto readFrameTime = 2 * tickLength;

  {
    int numCachedFrames = 1;

    auto potBufReader = Core::BufferedReader::create(std::make_shared<MockReader>(readFrameTime), numCachedFrames);
    ENSURE(potBufReader.status(), "BufferedReader should have been created correctly");
    auto bufReader = potBufReader.object();

    auto frame0 = bufReader->load();

    // return buffer triggers a reload
    std::thread readingThread([&]() { bufReader->releaseBuffer(frame0.buffer); });
    readingThread.detach();

    std::this_thread::sleep_for(tickLength);
    // destroy buffered reader
  }

  std::this_thread::sleep_for(2 * readFrameTime);
}

// make sure the Status from the reader is passed on frame by frame
// and not consumed in the buffering process
void testReaderStatus() {
  int numCachedFrames = 5;

  auto potBufReader = Core::BufferedReader::create(std::make_shared<StatusReader>(), numCachedFrames);
  ENSURE(potBufReader.status(), "BufferedReader should have been created correctly");
  auto bufReader = potBufReader.object();

  int readFrameStatus = 0;
  int seekFrameStatus = 0;

  for (int i = 0; i < 10; i++) {
    auto frame = bufReader->load();
    ENSURE_EQ(frame.readerStatus.getStatus().getErrorMessage(), std::to_string(readFrameStatus));
    auto reloadFrame = bufReader->reload();
    ENSURE_EQ(reloadFrame.readerStatus.getStatus().getErrorMessage(), std::to_string(readFrameStatus));
    bufReader->releaseBuffer(frame.buffer);
    bufReader->releaseBuffer(reloadFrame.buffer);
    readFrameStatus++;
  }

  srand(42);
  for (int i = 0; i < 10; i++) {
    ENSURE_EQ(bufReader->seekFrame(rand() % lastFrameMock).getErrorMessage(), std::to_string(seekFrameStatus++));
    readFrameStatus = 0;
    for (int j = 0; j < i; j++) {
      auto frame = bufReader->load();
      ENSURE_EQ(frame.readerStatus.getStatus().getErrorMessage(), std::to_string(readFrameStatus));
      auto reloadFrame = bufReader->reload();
      ENSURE_EQ(reloadFrame.readerStatus.getStatus().getErrorMessage(), std::to_string(readFrameStatus));
      bufReader->releaseBuffer(frame.buffer);
      bufReader->releaseBuffer(reloadFrame.buffer);
      readFrameStatus++;
    }
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  // test timeout 10 seconds
  // if there's a bug in the bufferedReader, it might hang indefinitely
  VideoStitch::Testing::initTestWithTimeoutInSeconds(30);

  auto log = VideoStitch::Logger::get(VideoStitch::Logger::Info);

  log << "Running testBufferedReaderSetup..." << std::endl;
  VideoStitch::Testing::testBufferedReaderSetup();

  for (int numCache : {0, 1, 2, 3, 5, 8}) {
    log << "Running testReaderBuffering(" << numCache << ")..." << std::endl;
    VideoStitch::Testing::testReaderBuffering(numCache);

    log << "Running testRaceConditions(" << numCache << ")..." << std::endl;
    VideoStitch::Testing::testRaceConditions(numCache);
  }

  log << "Running testBufferedReaderReload..." << std::endl;
  VideoStitch::Testing::testBufferedReaderReload();

  log << "Running testReturnOnThread..." << std::endl;
  VideoStitch::Testing::testReturnOnThread();

  log << "Running testLoadAfterSeekBlocking..." << std::endl;
  VideoStitch::Testing::testLoadAfterSeekBlocking();

  log << "Running testLoadAfterSeek..." << std::endl;
  VideoStitch::Testing::testLoadAfterSeek();

  log << "Running testDestroyWhileSeeking" << std::endl;
  VideoStitch::Testing::testDestroyWhileSeeking();

  log << "Running testDestroyWhileReading" << std::endl;
  VideoStitch::Testing::testDestroyWhileReading();

  log << "Running testReaderStatus..." << std::endl;
  VideoStitch::Testing::testReaderStatus();

  return 0;
}
