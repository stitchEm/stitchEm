// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/statefulReader.hxx"

#include "xiCamera.h"

#include <condition_variable>
#include <mutex>

/**
 * Ximea reader.
 */
namespace VideoStitch {
namespace Input {

class XimeaReader : public StatefulReader<unsigned char*> {
 public:
  virtual ~XimeaReader();

  static XimeaReader* create(const Ptv::Value* config, const int64_t width, const int64_t height);
  static bool handles(const Ptv::Value* config);

  virtual Status readFrame(int& frameId, unsigned char* data, Audio::Samples& audio);
  virtual Status readFrameAudioOnly(Audio::Samples& audio);
  virtual Status seekFrame(unsigned targetFrame);

  virtual Status unpackDevBuffer(const GPU::Buffer<uint32_t>& dst, const GPU::Buffer<const unsigned char>& src,
                                 GPU::Stream& stream) const;

  virtual Status perThreadInit();
  virtual void perThreadCleanup();

  virtual int getIntegerValue(const char* key, int defaultValue) const;
  virtual void setIntegerValue(const char* key, int value);

 private:
  XimeaReader(const int64_t width, const int64_t height, double fps, int64_t frameDataSize, std::string name,
              xi4Camera* device, int timeout);
  XimeaReader& operator=(const XimeaReader&);

  const std::string name;
  xi4Camera* device;
  int timeout;
  std::mutex mutex;
  int64_t currFrame;
};

}  // namespace Input
}  // namespace VideoStitch
