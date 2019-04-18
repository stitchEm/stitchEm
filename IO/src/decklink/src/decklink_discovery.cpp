// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "decklink_discovery.hpp"
#include "decklink_helpers.hpp"

#include "libvideostitch/logging.hpp"

#include <algorithm>
#include <map>

using namespace VideoStitch;
using namespace Plugin;

namespace {
std::vector<DisplayMode> getSupportedDisplayModes(IDeckLinkDisplayModeIterator* displayModeIterator) {
  std::vector<DisplayMode> supportedDisplayModes;
  IDeckLinkDisplayMode* displayMode = nullptr;
  while (displayModeIterator->Next(&displayMode) == S_OK) {
    BMDTimeValue frameRateDuration, frameRateScale;
    displayMode->GetFrameRate(&frameRateDuration, &frameRateScale);
    DisplayMode display_mode(
        displayMode->GetWidth(), displayMode->GetHeight(),
        displayMode->GetFieldDominance() == _BMDFieldDominance::bmdLowerFieldFirst ||
            displayMode->GetFieldDominance() == _BMDFieldDominance::bmdUpperFieldFirst,
        {(int)frameRateScale,
         (int)frameRateDuration});  // Rounding the framerate value given the frameScale and the frameDuration
    supportedDisplayModes.push_back(display_mode);
    displayMode->Release();
  }
  return supportedDisplayModes;
}

void filterDisplayModes(std::vector<DisplayMode>& displayModes) {
  // Filter not working display modes for output
  for (int index = 0; index < displayModes.size();) {
    const auto& displayMode = displayModes.at(index);
    bool widthIsTooSmall = displayMode.width <= 720;
    bool isDci = displayMode.width == 2048 && displayMode.height == 1080 ||
                 displayMode.width == 4096 && displayMode.height == 2160;
    if (widthIsTooSmall || isDci) {
      displayModes.erase(displayModes.begin() + index);
    } else {
      ++index;
    }
  }
}
}  // namespace

// ------------------------ Lifecycle --------------------------------

std::shared_ptr<DeckLinkDiscovery::Device> DeckLinkDiscovery::createDevice(std::shared_ptr<IDeckLink> deckLinkDevice,
                                                                           const std::string& deviceIdName,
                                                                           const std::string& deviceDisplayableName) {
  Plugin::DiscoveryDevice::Type type = Plugin::DiscoveryDevice::UNKNOWN;

  // Find the associated input device
  IDeckLinkInput* tempDeckLinkInput = nullptr;
  if (FAILED(deckLinkDevice->QueryInterface(IID_IDeckLinkInput, (void**)&tempDeckLinkInput))) {
    Logger::get(Logger::Info) << "DeckLink API: the " << deviceIdName
                              << " device does not accept input streaming capture." << std::endl;
  } else {
    type = static_cast<Plugin::DiscoveryDevice::Type>(type | Plugin::DiscoveryDevice::CAPTURE);
  }
  std::shared_ptr<IDeckLinkInput> deckLinkInput(tempDeckLinkInput, VideoStitch::DeckLink::getDefaultDeleter());

  // Find the associated output device
  IDeckLinkOutput* tempDeckLinkOutput = nullptr;
  if (FAILED(deckLinkDevice->QueryInterface(IID_IDeckLinkOutput, (void**)&tempDeckLinkOutput))) {
    Logger::get(Logger::Info) << "DeckLink API: the " << deviceIdName << " device can not output frames." << std::endl;
  } else {
    type = static_cast<Plugin::DiscoveryDevice::Type>(type | Plugin::DiscoveryDevice::PLAYBACK);
  }
  std::shared_ptr<IDeckLinkOutput> deckLinkOutput(tempDeckLinkOutput, VideoStitch::DeckLink::getDefaultDeleter());

  if (type == Plugin::DiscoveryDevice::UNKNOWN) {
    return std::shared_ptr<Device>();
  }

  // Configure
  std::shared_ptr<Device> device = std::make_shared<Device>();
  device->deckLinkDevice = deckLinkDevice;
  device->pluginDevice.type = type;
  device->pluginDevice.mediaType = Plugin::DiscoveryDevice::MediaType::VIDEO;
  device->pluginDevice.name = deviceIdName;
  device->pluginDevice.displayName = deviceDisplayableName;

  // Supported display modes
  std::vector<DisplayMode> displayModes;
  if (deckLinkInput) {
    IDeckLinkDisplayModeIterator* displayModeIterator = nullptr;
    if (FAILED(deckLinkInput->GetDisplayModeIterator(&displayModeIterator))) {
      Logger::get(Logger::Error) << "Error! DeckLink API could not to obtain the DisplayModeIterator interface on the "
                                 << device->pluginDevice.name << " input device." << std::endl;
      return std::shared_ptr<Device>();
    }
    displayModes = getSupportedDisplayModes(displayModeIterator);
  } else if (deckLinkOutput) {
    IDeckLinkDisplayModeIterator* displayModeIterator = nullptr;
    if (FAILED(deckLinkOutput->GetDisplayModeIterator(&displayModeIterator))) {
      Logger::get(Logger::Error) << "Error! DeckLink API could not to obtain the DisplayModeIterator interface on the "
                                 << device->pluginDevice.name << " output device." << std::endl;
      return std::shared_ptr<Device>();
    }
    displayModes = getSupportedDisplayModes(displayModeIterator);
  }
  filterDisplayModes(displayModes);
  device->supportedDisplayModes = displayModes;

  // Supported pixel formats
  for (const DeckLink::Helpers::PixelFormatStruct& pfs : DeckLink::Helpers::getInstance().commonPixelFormats()) {
    for (const auto& displayMode : device->supportedDisplayModes) {
      BMDDisplayMode bmdDisplayMode = DeckLink::Helpers::getInstance().bmdDisplayMode(displayMode);
      BMDDisplayModeSupport displayModeSupport;

      if (deckLinkInput) {
        if (SUCCEEDED(deckLinkInput->DoesSupportVideoMode(bmdDisplayMode, pfs.bmdPixelFormat, bmdVideoInputFlagDefault,
                                                          &displayModeSupport, NULL)) &&
            displayModeSupport != bmdDisplayModeNotSupported) {
          device->supportedPixelFormats.push_back(pfs.pixelFormat);
          break;
        }
      } else if (deckLinkOutput) {
        if (SUCCEEDED(deckLinkOutput->DoesSupportVideoMode(bmdDisplayMode, pfs.bmdPixelFormat,
                                                           bmdVideoOutputFlagDefault, &displayModeSupport, NULL)) &&
            displayModeSupport != bmdDisplayModeNotSupported) {
          device->supportedPixelFormats.push_back(pfs.pixelFormat);
          break;
        }
      }
    }
  }

  return device;
}

DeckLinkDiscovery* DeckLinkDiscovery::create() {
  // Retrieve the devices
  std::vector<std::string> cards = VideoStitch::DeckLink::retrieveCardsNames();
  auto names = VideoStitch::DeckLink::retrieveDevicesNames();  // list of <device id name, device displayable name>
  int nameIndex = 0;
  std::vector<std::shared_ptr<Device>> devices;
  IDeckLink* tempDeckLinkDevice = nullptr;
  std::shared_ptr<IDeckLinkIterator> decklinkIterator = VideoStitch::DeckLink::createIterator();

  while (decklinkIterator && decklinkIterator->Next(&tempDeckLinkDevice) == S_OK && names.size()) {
    std::shared_ptr<IDeckLink> deckLinkDevice(tempDeckLinkDevice, VideoStitch::DeckLink::getDefaultDeleter());
    std::string deviceIdName = names.at(nameIndex).first;
    std::string deviceDisplayableName = names.at(nameIndex).second;
    ++nameIndex;
    if (deviceIdName.empty()) {
      continue;
    }
    Logger::get(Logger::Debug) << "DeckLink API: found a device: " << deviceIdName << "." << std::endl;

    std::shared_ptr<Device> device = createDevice(deckLinkDevice, deviceIdName, deviceDisplayableName);
    if (device) {
      devices.push_back(device);
    }
  }

  std::sort(devices.begin(), devices.end(), [](std::shared_ptr<Device> lhs, std::shared_ptr<Device> rhs) -> bool {
    return lhs->pluginDevice.displayName < rhs->pluginDevice.displayName;
  });
  return new DeckLinkDiscovery(cards, devices);
}

DeckLinkDiscovery::DeckLinkDiscovery(const std::vector<std::string>& cards,
                                     const std::vector<std::shared_ptr<Device>>& devices)
    : m_cards(cards), m_devices(devices) {}

DeckLinkDiscovery::Device::~Device() {}

// -------------------------- Plugin implementation ----------------------------------

std::string DeckLinkDiscovery::name() const { return "decklink"; }

std::string DeckLinkDiscovery::readableName() const { return "Blackmagic DeckLink"; }

std::vector<Plugin::DiscoveryDevice> DeckLinkDiscovery::inputDevices() {
  std::vector<Plugin::DiscoveryDevice> devices;
  std::for_each(m_devices.begin(), m_devices.end(), [&devices](const std::shared_ptr<Device>& device) {
    if (device->pluginDevice.type & Plugin::DiscoveryDevice::CAPTURE) {
      devices.push_back(device->pluginDevice);
    }
  });
  return devices;
}

std::vector<Plugin::DiscoveryDevice> DeckLinkDiscovery::outputDevices() {
  std::vector<Plugin::DiscoveryDevice> devices;
  std::for_each(m_devices.begin(), m_devices.end(), [&devices](const std::shared_ptr<Device>& device) {
    if (device->pluginDevice.type & Plugin::DiscoveryDevice::PLAYBACK) {
      devices.push_back(device->pluginDevice);
    }
  });
  return devices;
}

std::vector<std::string> DeckLinkDiscovery::cards() const { return m_cards; }

void DeckLinkDiscovery::registerAutoDetectionCallback(AutoDetection& autoDetection) {
  return;  // Incompatible with DeckLink Reader in its actual form, need a wrapper of the DeckLink SDK
}

std::vector<DisplayMode> DeckLinkDiscovery::supportedDisplayModes(const Plugin::DiscoveryDevice& device) {
  auto it = std::find_if(
      m_devices.begin(), m_devices.end(),
      [&device](const std::shared_ptr<Device>& m_device) -> bool { return device == m_device->pluginDevice; });
  if (it != m_devices.end()) {
    return (*it)->supportedDisplayModes;
  } else {
    return std::vector<DisplayMode>();
  }
}

std::vector<PixelFormat> DeckLinkDiscovery::supportedPixelFormat(const Plugin::DiscoveryDevice& device) {
  auto it = std::find_if(
      m_devices.begin(), m_devices.end(),
      [&device](const std::shared_ptr<Device>& m_device) -> bool { return device == m_device->pluginDevice; });
  if (it != m_devices.end()) {
    return (*it)->supportedPixelFormats;
  } else {
    return std::vector<PixelFormat>();
  }
}

std::vector<int> DeckLinkDiscovery::supportedNbChannels(const Plugin::DiscoveryDevice& device) {
  std::vector<int> channels;
  channels.push_back(2);
  channels.push_back(8);
  channels.push_back(16);
  return channels;
}

std::vector<Audio::SamplingRate> DeckLinkDiscovery::supportedSamplingRates(const Plugin::DiscoveryDevice& device) {
  std::vector<Audio::SamplingRate> rates;
  rates.push_back(Audio::SamplingRate::SR_48000);
  return rates;
}

std::vector<Audio::SamplingDepth> DeckLinkDiscovery::supportedSampleFormats(const Plugin::DiscoveryDevice& device) {
  std::vector<Audio::SamplingDepth> formats;
  formats.push_back(Audio::SamplingDepth::INT16);
  formats.push_back(Audio::SamplingDepth::INT32);
  return formats;
}
