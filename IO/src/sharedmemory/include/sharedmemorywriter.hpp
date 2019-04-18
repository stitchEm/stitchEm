// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/stitchOutput.hpp"

#include <QtCore/QtGlobal>

#include <memory>

class QSharedMemory;
class QString;

namespace VideoStitch {
namespace Output {

class SharedMemoryWriter : public Output {
 public:
  static std::string getTypeName();
  static bool handles(const VideoStitch::Ptv::Value* config);
  static VideoStitch::Output::Output* create(const VideoStitch::Ptv::Value* config, const std::string& name,
                                             unsigned width, unsigned height, int paddingTop, int paddingBottom);

 public:
  ~SharedMemoryWriter();

  void pushVideo(mtime_t date, const char* videoFrame) override;
  void pushAudio(Audio::Samples& audioSamples) override {}

 private:
  SharedMemoryWriter(const std::string& name, unsigned width, unsigned height, int paddingTop, int paddingBottom,
                     QString key, quint8 newNumberOfFrames, int numberOfCroppedLines = 0);
  void writeBufferHeader();
  void writeAllFrameHeaders();
  quint32 getTrueHeight() const;
  quint32 getBytePerRow() const;
  quint32 getImageSize() const;
  quint32 getFrameSize() const;
  int getSharedMemorySize() const;

  std::unique_ptr<QSharedMemory> sharedMemory;
  quint8 counter;
  const quint8 numberOfFrames;
  const int numberOfCroppedLines;

  static const quint8 bufferProtocolVersion;
  static const quint8 frameProtocolVersion;
  static const quint8 bufferHeaderSize;
  static const quint16 frameHeaderSize;
  static const quint8 defaultNumberOfFrames;
  static const quint8 imageFormat;
  static const VideoStitch::PixelFormat pixelFormat;
  static const quint32 bytePerPixel;
  // BUFFER_HEADER
  static const int position_counter;
  static const int position_bufferProtocolVersion;
  static const int position_bufferHeaderSize;
  static const int position_numberOfFrames;
  static const int position_frameSize;
  static const int position_mostRecentFrameIndex;
  // FRAME_HEADER
  static const int position_frameProtocolVersion;
  static const int position_frameHeaderSize;
  static const int position_isValid;
  static const int position_width;
  static const int position_height;
  static const int position_bytePerRow;
  static const int position_imageFormat;
  static const int position_frameCounter;
  static const int position_timestamp;
};
}  // namespace Output
}  // namespace VideoStitch
