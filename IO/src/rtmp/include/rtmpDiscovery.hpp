// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/plugin.hpp"

#include <vector>
#include <string>

namespace VideoStitch {
namespace Plugin {

class RTMPDiscovery : public VSDiscoveryPlugin {
 public:
  static RTMPDiscovery* create();
  virtual ~RTMPDiscovery();

  virtual std::string name() const override;
  virtual std::string readableName() const override;
  virtual std::vector<Plugin::DiscoveryDevice> outputDevices() override;
  virtual std::vector<Plugin::DiscoveryDevice> inputDevices() override;
  virtual std::vector<std::string> supportedVideoCodecs() override;
  virtual std::vector<PixelFormat> supportedPixelFormat(const Plugin::DiscoveryDevice&) override {
    return std::vector<PixelFormat>();
  }
  virtual std::vector<int> supportedNbChannels(const Plugin::DiscoveryDevice&) override { return std::vector<int>(); }
  virtual std::vector<Audio::SamplingRate> supportedSamplingRates(const Plugin::DiscoveryDevice&) override {
    return std::vector<Audio::SamplingRate>();
  }
  virtual std::vector<Audio::SamplingDepth> supportedSampleFormats(const Plugin::DiscoveryDevice&) override {
    return std::vector<Audio::SamplingDepth>();
  }

  virtual std::vector<std::string> cards() const override { return std::vector<std::string>(); }
  virtual std::vector<DisplayMode> supportedDisplayModes(const Plugin::DiscoveryDevice&) override {
    return std::vector<DisplayMode>();
  }
  virtual DisplayMode currentDisplayMode(const Plugin::DiscoveryDevice&) override { return DisplayMode(); }
  virtual void registerAutoDetectionCallback(AutoDetection&) override {}

 private:
  RTMPDiscovery();

  std::vector<std::string> m_supportedCodecs;
};

}  // namespace Plugin
}  // namespace VideoStitch
