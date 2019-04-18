// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/plugin.hpp"

#include "ntv2Helper.hpp"

#include <vector>
#include <string>
#include <memory>

#include <windows.h>

#include "ajatypes.h"
#include "ajastuff/common/types.h"
#include "ntv2card.h"
#include "ntv2devicescanner.h"
#include "ntv2publicinterface.h"

namespace VideoStitch {
namespace Plugin {

class Ntv2Discovery : public VSDiscoveryPlugin {
  struct Device {
    Device() : boardIdx(0), channelIdx(0), autodetection(nullptr) {}

    Plugin::DiscoveryDevice pluginDevice;
    NTV2DeviceInfo boardInfo;
    uint32_t boardIdx;
    uint32_t channelIdx;

    AutoDetection* autodetection;
  };

  struct InputDevice : public Device {
    InputDevice() : Device() {}
  };

  struct OutputDevice : public Device {
    OutputDevice() : Device() {}
  };

 public:
  static Ntv2Discovery* create();
  virtual ~Ntv2Discovery();

  virtual std::string name() const override;
  virtual std::string readableName() const override;
  virtual std::vector<Plugin::DiscoveryDevice> inputDevices() override;
  virtual std::vector<Plugin::DiscoveryDevice> outputDevices() override;
  virtual std::vector<std::string> cards() const override;
  virtual void registerAutoDetectionCallback(AutoDetection&) override;
  virtual std::vector<DisplayMode> supportedDisplayModes(const Plugin::DiscoveryDevice&) override;
  virtual std::vector<PixelFormat> supportedPixelFormat(const Plugin::DiscoveryDevice&) override;
  virtual std::vector<int> supportedNbChannels(const Plugin::DiscoveryDevice& device) override;
  virtual std::vector<Audio::SamplingRate> supportedSamplingRates(const Plugin::DiscoveryDevice& device) override;
  virtual std::vector<Audio::SamplingDepth> supportedSampleFormats(const Plugin::DiscoveryDevice& device) override;

  bool supportVideoMode(const Plugin::DiscoveryDevice&, const DisplayMode&, const PixelFormat&);

 private:
  Ntv2Discovery(const std::vector<std::string>& cards, const std::vector<std::shared_ptr<Device>>& devices);

  DisplayMode currentDisplayMode(const Plugin::DiscoveryDevice& device);
  static Audio::SamplingRate convertSamplerate(AudioSampleRateEnum ntv2SampleRate);
  static Audio::SamplingDepth convertFormats(AudioBitsPerSampleEnum ntv2Format);

  std::vector<std::string> m_cards;
  std::vector<std::shared_ptr<Device>> m_devices;
};

}  // namespace Plugin
}  // namespace VideoStitch
