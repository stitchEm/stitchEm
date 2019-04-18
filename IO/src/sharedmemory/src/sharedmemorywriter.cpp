// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "sharedmemorywriter.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/stitchOutput.hpp"

#include <QtCore/QSharedMemory>

namespace VideoStitch {
namespace Output {

// To understand the data structure, cf \\NAS_VS\Assets\Customers\Rheinmetall\Shared Memory Interface.pdf
const quint8 SharedMemoryWriter::bufferProtocolVersion = 1u;
const quint8 SharedMemoryWriter::frameProtocolVersion = 1u;
const quint8 SharedMemoryWriter::bufferHeaderSize = 16u;  // We use 16 bytes instead of 9 for data alignment
const quint16 SharedMemoryWriter::frameHeaderSize = 48u;  // We use 48 bytes instead of 43 for data alignment
const quint8 SharedMemoryWriter::defaultNumberOfFrames = 8u;
const VideoStitch::PixelFormat SharedMemoryWriter::pixelFormat = VideoStitch::PixelFormat::RGB;
const quint8 SharedMemoryWriter::imageFormat =
    3u;  // 3 is the code for VideoStitch::PixelFormat::RGB in the Shared Memory Interface specifications
const quint32 SharedMemoryWriter::bytePerPixel = 3u;  // For VideoStitch::PixelFormat::RGB
// BUFFER_HEADER
const int SharedMemoryWriter::position_counter = 0;
const int SharedMemoryWriter::position_bufferProtocolVersion = 1;
const int SharedMemoryWriter::position_bufferHeaderSize = 2;
const int SharedMemoryWriter::position_numberOfFrames = 3;
const int SharedMemoryWriter::position_frameSize = 4;
const int SharedMemoryWriter::position_mostRecentFrameIndex = 8;
// FRAME_HEADER
const int SharedMemoryWriter::position_frameProtocolVersion = 0;
const int SharedMemoryWriter::position_frameHeaderSize = 1;
const int SharedMemoryWriter::position_isValid = 3;
const int SharedMemoryWriter::position_width = 4;
const int SharedMemoryWriter::position_height = 6;
const int SharedMemoryWriter::position_bytePerRow = 8;
const int SharedMemoryWriter::position_imageFormat = 10;
const int SharedMemoryWriter::position_frameCounter = 11;
const int SharedMemoryWriter::position_timestamp = 19;

std::string SharedMemoryWriter::getTypeName() { return "shared_memory"; }

bool SharedMemoryWriter::handles(const VideoStitch::Ptv::Value* config) {
  return config && config->has("type") && config->has("type")->asString() == getTypeName() && config->has("filename") &&
         !config->has("filename")->asString().empty();
}

VideoStitch::Output::Output* SharedMemoryWriter::create(const VideoStitch::Ptv::Value* config, const std::string& name,
                                                        unsigned width, unsigned height, int paddingTop,
                                                        int paddingBottom) {
  std::string key;
  if (Parse::populateString("Shared memory writer", *config, "key", key, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "Error! Shared memory key (\"key\") couldn't be retrieved. Please, put a key in your "
                                  "configuration. Aborting."
                               << std::endl;
    return nullptr;
  }

  int numberOfFrames = defaultNumberOfFrames;
  if (Parse::populateInt("Shared memory writer", *config, "number_of_frames", numberOfFrames, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Warning)
        << "Warning! Shared memory number of frames (\"number_of_frames\") couldn't be retrieved. Use "
        << defaultNumberOfFrames << " by default." << std::endl;
  }
  if (numberOfFrames < 0) {
    numberOfFrames = defaultNumberOfFrames;
    Logger::get(Logger::Warning)
        << "Warning! Shared memory number of frames (\"number_of_frames\") is negative. Set to "
        << defaultNumberOfFrames << " instead." << std::endl;
  }

  int numberOfCroppedLines = 0;
  Parse::populateInt("Shared memory writer", *config, "crop_bottom", numberOfCroppedLines, false);
  // No log if the value is not present because it's normal to not crop
  if (numberOfCroppedLines < 0) {
    numberOfCroppedLines = 0;
    Logger::get(Logger::Warning)
        << "Warning! Shared memory number of cropped lines (\"crop_bottom\") is negative. Set to 0 instead."
        << std::endl;
  }

  Logger::get(Logger::Info) << "Create a shared memory writer" << std::endl;
  return new SharedMemoryWriter(name, width, height, paddingTop, paddingBottom, QString::fromStdString(key),
                                quint8(numberOfFrames), numberOfCroppedLines);
}

SharedMemoryWriter::SharedMemoryWriter(const std::string& name, unsigned width, unsigned height, int paddingTop,
                                       int paddingBottom, QString key, quint8 newNumberOfFrames,
                                       int newNumberOfCroppedLines)
    : Output(name),
      VideoWriter(width, height, {-1, 1}, paddingTop, paddingBottom, pixelFormat, Host),
      sharedMemory(new QSharedMemory()),
      counter(0u),
      numberOfFrames(newNumberOfFrames),
      numberOfCroppedLines(newNumberOfCroppedLines) {
  sharedMemory->setNativeKey(key);

  if (!sharedMemory->create(getSharedMemorySize())) {
    Logger::get(Logger::Error) << "Error when creating shared memory: " << sharedMemory->errorString().toStdString()
                               << std::endl;
  } else {
    writeBufferHeader();
    writeAllFrameHeaders();
  }
}

SharedMemoryWriter::~SharedMemoryWriter() {}

void SharedMemoryWriter::pushVideo(mtime_t date, const char* videoFrame) {
  if (!videoFrame || !sharedMemory->isAttached()) {
    return;
  }

  // To understand the data structure, cf \\NAS_VS\Assets\Customers\Rheinmetall\Shared Memory Interface.pdf
  quint8* sharedMemoryPtr = (quint8*)sharedMemory->data();
  quint8 newMostRecentFrameIndex = (*(sharedMemoryPtr + position_mostRecentFrameIndex) + 1u) % numberOfFrames;
  const quint32 frameSize = *((quint32*)(sharedMemoryPtr + position_frameSize));
  quint8* framePtr = sharedMemoryPtr + bufferHeaderSize + newMostRecentFrameIndex * frameSize;

  *(framePtr + position_isValid) = quint8(1u);  // Update isValid
  *((quint64*)(framePtr + position_frameCounter)) =
      quint64(date * (double)(getFrameRate().num) / ((double)(getFrameRate().den) * 1000000.0));  // Update frameCounter
  auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();
  *((double*)(framePtr + position_timestamp)) = double(milliseconds);  // Update timestamp

  quint8* imageDataPtr = framePtr + frameHeaderSize;
  memcpy((void*)(imageDataPtr), (const void*)(videoFrame), getImageSize());  // Update image data

  *(sharedMemoryPtr + position_mostRecentFrameIndex) = newMostRecentFrameIndex;  // Update the mostRecentFrameIndex
  *sharedMemoryPtr = ++counter;  // Updating the counter should be the last operation
}

void SharedMemoryWriter::writeBufferHeader() {
  // To understand the data structure, cf \\NAS_VS\Assets\Customers\Rheinmetall\Shared Memory Interface.pdf
  quint8* ptr = (quint8*)sharedMemory->data();
  *(ptr + position_counter) = counter;                              // Initialize counter
  *(ptr + position_bufferProtocolVersion) = bufferProtocolVersion;  // Initialize bufferProtocolVersion
  *(ptr + position_bufferHeaderSize) = bufferHeaderSize;            // Initialize bufferHeaderSize
  *(ptr + position_numberOfFrames) = numberOfFrames;                // Initialize numberOfFrames
  *((quint32*)(ptr + position_frameSize)) = getFrameSize();         // Initialize frameSize
  *(ptr + position_mostRecentFrameIndex) =
      quint8(numberOfFrames - 1u);  // Initialize mostRecentFrameIndex (the first valid frame will be 0)
}

void SharedMemoryWriter::writeAllFrameHeaders() {
  // To understand the data structure, cf \\NAS_VS\Assets\Customers\Rheinmetall\Shared Memory Interface.pdf
  quint8* framesPtr = (quint8*)sharedMemory->data() + bufferHeaderSize;

  for (quint8 indexOfFrame = 0u; indexOfFrame < numberOfFrames; ++indexOfFrame) {
    quint8* frameBeginningPtr = framesPtr + getFrameSize() * quint32(indexOfFrame);
    *(frameBeginningPtr + position_frameProtocolVersion) = frameProtocolVersion;    // Initialize frameProtocolVersion
    *((quint16*)(frameBeginningPtr + position_frameHeaderSize)) = frameHeaderSize;  // Initialize frameHeaderSize
    *(frameBeginningPtr + position_isValid) = quint8(0u);                      // Initialize isValid, invalid by default
    *((quint16*)(frameBeginningPtr + position_width)) = getWidth();            // Initialize width
    *((quint16*)(frameBeginningPtr + position_height)) = getTrueHeight();      // Initialize height
    *((quint16*)(frameBeginningPtr + position_bytePerRow)) = getBytePerRow();  // Initialize bytePerRow
    *(frameBeginningPtr + position_imageFormat) = imageFormat;                 // Initialize imageFormat
    *((quint64*)(frameBeginningPtr + position_frameCounter)) = quint64(0u);    // Initialize frameCounter
    *((double*)(frameBeginningPtr + position_timestamp)) = 0.0;                // Initialize timestamp
  }
}

quint32 SharedMemoryWriter::getTrueHeight() const {
  return getHeight() - getPaddingTop() - getPaddingBottom() - numberOfCroppedLines;
}

quint32 SharedMemoryWriter::getBytePerRow() const { return getWidth() * bytePerPixel; }

quint32 SharedMemoryWriter::getImageSize() const { return getTrueHeight() * getBytePerRow(); }

quint32 SharedMemoryWriter::getFrameSize() const { return quint32(frameHeaderSize) + getImageSize(); }

int SharedMemoryWriter::getSharedMemorySize() const { return bufferHeaderSize + numberOfFrames * getFrameSize(); }

}  // namespace Output
}  // namespace VideoStitch
