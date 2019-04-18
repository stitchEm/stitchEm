// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/input.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/plugin.hpp"

#include <linux/videodev2.h>

/**
 * IO plugin based on the Video for Linux API
 *
 * This should be able to control every capture
 * card for which the manufacturer provided
 * a v4l2 compliant driver.
 * Tested with a Magewell ProCapture.
 *
 * This code borrows heavily from
 * http://linuxtv.org/downloads/v4l-dvb-apis/capture-example.html
 **/

namespace VideoStitch {
namespace Input {

struct buffer_t {
  void* start;
  size_t length;
};

class V4L2Reader : public VideoReader {
 public:
  V4L2Reader(int id, int fd, buffer_t* bufv, uint32_t bufc, int64_t width, int64_t height, int64_t frameDataSize,
             VideoStitch::PixelFormat format, FrameRate frameRate);
  ~V4L2Reader();

  static V4L2Reader* create(const Ptv::Value* config, const Plugin::VSReaderPlugin::Config& runtime);
  static bool handles(const Ptv::Value* config);

  ReadStatus readFrame(mtime_t&, unsigned char*);
  Status seekFrame(frameid_t);

 private:
  static buffer_t* startMmap(int fd, uint32_t* n);
  static void stopMmap(int fd, buffer_t* bufv, uint32_t bufc);

  mtime_t getFrameTimestamp(const v4l2_buffer*);
  static int findMaxRate(int fd, v4l2_format fmt, const v4l2_fract* min_it, v4l2_fract* it);

  int fd;
  buffer_t* bufv;
  uint32_t bufc;
};
}  // namespace Input

namespace Plugin {

class V4L2Discovery : public VSDiscoveryPlugin {
 public:
  static V4L2Discovery* create();
  virtual ~V4L2Discovery() {}

  virtual std::string name() const;
  virtual std::string readableName() const;
  virtual std::vector<Plugin::DiscoveryDevice> inputDevices();
  virtual std::vector<Plugin::DiscoveryDevice> outputDevices() { return std::vector<Plugin::DiscoveryDevice>(); }
  virtual std::vector<std::string> cards() const;

  virtual void registerAutoDetectionCallback(AutoDetection&) {}

  virtual std::vector<DisplayMode> supportedDisplayModes(const Plugin::DiscoveryDevice&) {
    return std::vector<DisplayMode>();
  }
  DisplayMode currentDisplayMode(const Plugin::DiscoveryDevice&) {
    return DisplayMode();  // TODO
  }
  virtual std::vector<PixelFormat> supportedPixelFormat(const Plugin::DiscoveryDevice&) {
    return std::vector<PixelFormat>();
  }
  virtual std::vector<int> supportedNbChannels(const Plugin::DiscoveryDevice&) { return std::vector<int>(); }
  virtual std::vector<Audio::SamplingRate> supportedSamplingRates(const Plugin::DiscoveryDevice&) {
    return std::vector<Audio::SamplingRate>();
  }
  virtual std::vector<Audio::SamplingDepth> supportedSampleFormats(const Plugin::DiscoveryDevice&) {
    return std::vector<Audio::SamplingDepth>();
  }

 private:
  V4L2Discovery() {}
};

}  // namespace Plugin
}  // namespace VideoStitch
