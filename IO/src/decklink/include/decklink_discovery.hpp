// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef DECKLINK_DISCOVERY_HPP_
#define DECKLINK_DISCOVERY_HPP_

#include "libvideostitch/plugin.hpp"

#if defined(_WIN32)
#include "DeckLinkAPI_h.h"
#else
#include "DeckLinkAPI.h"
#include "DeckLinkAPIModes.h"
#endif

#include <vector>
#include <string>
#include <memory>

namespace VideoStitch {
namespace Plugin {

class DeckLinkDiscovery : public VSDiscoveryPlugin {
  struct Device {
    Device() : deckLinkDevice(nullptr), supportedDisplayModes(), supportedPixelFormats() {
      pluginDevice.type = Plugin::DiscoveryDevice::UNKNOWN;
    }
    virtual ~Device();

    Plugin::DiscoveryDevice pluginDevice;
    std::shared_ptr<IDeckLink> deckLinkDevice;
    std::vector<DisplayMode> supportedDisplayModes;
    std::vector<PixelFormat> supportedPixelFormats;
  };

 public:
  static std::shared_ptr<Device> createDevice(std::shared_ptr<IDeckLink> deckLinkDevice,
                                              const std::string& deviceIdName,
                                              const std::string& deviceDisplayableName);
  static DeckLinkDiscovery* create();
  virtual ~DeckLinkDiscovery() {}

  virtual std::string name() const;
  virtual std::string readableName() const;

  virtual std::vector<Plugin::DiscoveryDevice> inputDevices();
  virtual std::vector<Plugin::DiscoveryDevice> outputDevices();
  virtual std::vector<std::string> cards() const;

  virtual void registerAutoDetectionCallback(AutoDetection&);

  virtual std::vector<DisplayMode> supportedDisplayModes(const Plugin::DiscoveryDevice&);
  DisplayMode currentDisplayMode(const Plugin::DiscoveryDevice&) {
    return DisplayMode();  // TODO
  }
  virtual std::vector<PixelFormat> supportedPixelFormat(const Plugin::DiscoveryDevice&);
  virtual std::vector<int> supportedNbChannels(const Plugin::DiscoveryDevice& device);
  virtual std::vector<Audio::SamplingRate> supportedSamplingRates(const Plugin::DiscoveryDevice& device);
  virtual std::vector<Audio::SamplingDepth> supportedSampleFormats(const Plugin::DiscoveryDevice& device);

 private:
  DeckLinkDiscovery(const std::vector<std::string>& cards, const std::vector<std::shared_ptr<Device>>& devices);

  std::vector<std::string> m_cards;
  std::vector<std::shared_ptr<Device>> m_devices;
};

}  // namespace Plugin
}  // namespace VideoStitch
#endif  // DECKLINK_DISCOVERY_HPP_
