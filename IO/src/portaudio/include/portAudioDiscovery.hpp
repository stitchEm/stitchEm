// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/plugin.hpp"
#include <vector>

namespace VideoStitch {
namespace Plugin {

class PortAudioDiscovery : public VSDiscoveryPlugin {
 public:
  static PortAudioDiscovery* create();
  virtual ~PortAudioDiscovery() {}

  std::string name() const override;
  std::string readableName() const override;
  std::vector<Plugin::DiscoveryDevice> inputDevices() override;
  std::vector<Plugin::DiscoveryDevice> outputDevices() override;
  std::vector<std::string> cards() const override;
  void registerAutoDetectionCallback(AutoDetection&) override;
  std::vector<DisplayMode> supportedDisplayModes(const Plugin::DiscoveryDevice&) override;
  DisplayMode currentDisplayMode(const Plugin::DiscoveryDevice&) override;
  std::vector<PixelFormat> supportedPixelFormat(const Plugin::DiscoveryDevice&) override;
  std::vector<int> supportedNbChannels(const Plugin::DiscoveryDevice& device) override;
  std::vector<Audio::SamplingRate> supportedSamplingRates(const Plugin::DiscoveryDevice& device) override;
  std::vector<Audio::SamplingDepth> supportedSampleFormats(const Plugin::DiscoveryDevice& device) override;

 private:
  PortAudioDiscovery();

  void filterWindowsDevices();
  void removeDevice(size_t index);
  std::vector<std::string> _devNames;

  std::map<std::string, int> _devMaxInputsChannel;
  std::map<std::string, int> _devMaxOutputsChannel;
  std::map<std::string, std::vector<int>> _devChannelCounts;
  std::map<std::string, std::vector<Audio::SamplingRate>> _devSampleRates;
  std::map<std::string, std::vector<Audio::SamplingDepth>> _devSampleDepths;
};

}  // namespace Plugin
}  // namespace VideoStitch
