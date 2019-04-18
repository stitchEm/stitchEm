// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ntv2Discovery.hpp"

#include "libvideostitch/logging.hpp"

#include <thread>
#include <future>
#include <chrono>
#include <algorithm>
#include <locale>
#include <codecvt>

#include <ntv2utils.h>

using namespace VideoStitch;
using namespace Plugin;

Ntv2Discovery* Ntv2Discovery::create() {
  std::vector<std::string> cards;
  std::vector<std::shared_ptr<Device>> devices;

  CNTV2DeviceScanner ajaDeviceScanner;
  ajaDeviceScanner.ScanHardware();
  size_t nbCard = ajaDeviceScanner.GetNumDevices();
  if (nbCard == 0) return nullptr;

  for (uint32_t iDevice = 0; iDevice < nbCard; ++iDevice) {
    Device device;
    device.boardInfo = ajaDeviceScanner.GetDeviceInfoList()[iDevice];
    CNTV2Card card;
    CNTV2DeviceScanner::GetDeviceAtIndex(iDevice, card);
    cards.push_back(device.boardInfo.deviceIdentifier);

    for (int8_t i = 0; i < device.boardInfo.numVidInputs; ++i) {
      std::shared_ptr<InputDevice> inputDevice = std::make_shared<InputDevice>();
      // Aja inputs are labeled from 1 to numVidInputs
      inputDevice->pluginDevice.displayName = device.boardInfo.deviceIdentifier + " Input " + std::to_string(i + 1);
      inputDevice->pluginDevice.name = std::to_string(device.boardInfo.deviceIndex) + std::to_string(i);
      inputDevice->pluginDevice.type = Plugin::DiscoveryDevice::CAPTURE;
      inputDevice->pluginDevice.mediaType = Plugin::DiscoveryDevice::MediaType::AUDIO_AND_VIDEO;
      inputDevice->boardIdx = device.boardInfo.deviceIndex;
      inputDevice->channelIdx = i;
      inputDevice->boardInfo = device.boardInfo;
      devices.push_back(inputDevice);
    }

    for (int8_t i = 0; i < device.boardInfo.numVidOutputs; ++i) {
      std::shared_ptr<OutputDevice> outputDevice = std::make_shared<OutputDevice>();
      // Aja outputs are labeled from 1 to numVidOutputs
      outputDevice->pluginDevice.displayName = device.boardInfo.deviceIdentifier + " Output " + std::to_string(i + 1);
      outputDevice->pluginDevice.name = std::to_string(device.boardInfo.deviceIndex) + std::to_string(i);
      outputDevice->pluginDevice.type = Plugin::DiscoveryDevice::PLAYBACK;
      outputDevice->pluginDevice.mediaType = Plugin::DiscoveryDevice::MediaType::AUDIO_AND_VIDEO;
      outputDevice->boardIdx = device.boardInfo.deviceIndex;
      outputDevice->channelIdx = i;
      outputDevice->boardInfo = device.boardInfo;
      devices.push_back(outputDevice);
    }
  }
  return new Ntv2Discovery(cards, devices);
}

Ntv2Discovery::Ntv2Discovery(const std::vector<std::string>& cards, const std::vector<std::shared_ptr<Device>>& devices)
    : m_cards(cards), m_devices(devices) {}

Ntv2Discovery::~Ntv2Discovery() {}

std::string Ntv2Discovery::name() const { return "aja"; }

std::string Ntv2Discovery::readableName() const { return "AJA"; }

std::vector<Plugin::DiscoveryDevice> Ntv2Discovery::inputDevices() {
  std::vector<Plugin::DiscoveryDevice> pluginDevices;

  for (auto it = m_devices.begin(); it != m_devices.end(); ++it) {
    if ((*it)->pluginDevice.type == Plugin::DiscoveryDevice::CAPTURE) {
      pluginDevices.push_back((*it)->pluginDevice);
    }
  }

  return pluginDevices;
}

std::vector<Plugin::DiscoveryDevice> Ntv2Discovery::outputDevices() {
  std::vector<Plugin::DiscoveryDevice> pluginDevices;

  for (auto it = m_devices.begin(); it != m_devices.end(); ++it) {
    if ((*it)->pluginDevice.type == Plugin::DiscoveryDevice::PLAYBACK) {
      pluginDevices.push_back((*it)->pluginDevice);
    }
  }

  return pluginDevices;
}

std::vector<string> Ntv2Discovery::cards() const { return m_cards; }

void Ntv2Discovery::registerAutoDetectionCallback(AutoDetection& autoDetection) {
  return;  // Incompatible with AJA Reader in its actual form, need a wrapper of the AJA SDK
}

DisplayMode Ntv2Discovery::currentDisplayMode(const DiscoveryDevice& device) {
  const auto it = std::find_if(
      m_devices.begin(), m_devices.end(),
      [&device](const std::shared_ptr<Device>& m_device) -> bool { return device == m_device->pluginDevice; });
  if (it != m_devices.end()) {
    CNTV2Card card;
    if (!CNTV2DeviceScanner::GetDeviceAtIndex((*it)->boardIdx, card)) {
      return DisplayMode(0, 0, false, {1, 1});
    }

    const NTV2InputSource inputSource = GetNTV2InputSourceForIndex((*it)->channelIdx);
    if (inputSource == NTV2_INPUTSOURCE_INVALID) {
      return DisplayMode(0, 0, false, {1, 1});
    }

    const NTV2VideoFormat videoFormat = card.GetInputVideoFormat(inputSource);
    return aja2vsDisplayFormat(videoFormat);
  }
  return DisplayMode(0, 0, false, {1, 1});
}

std::vector<DisplayMode> Ntv2Discovery::supportedDisplayModes(const Plugin::DiscoveryDevice& device) {
  auto it = std::find_if(
      m_devices.begin(), m_devices.end(),
      [&device](const std::shared_ptr<Device>& m_device) -> bool { return device == m_device->pluginDevice; });
  if (it != m_devices.end()) {
    std::vector<DisplayMode> supportedDisplayModes;
    CNTV2Card card;
    CNTV2DeviceScanner::GetDeviceAtIndex((*it)->boardIdx, card);
    NTV2VideoFormatSet outFormats;
    card.GetSupportedVideoFormats(outFormats);

    for (auto it = outFormats.begin(); it != outFormats.end(); ++it) {
      const DisplayMode displayMode = aja2vsDisplayFormat((*it));
      if (displayMode.width != 0 && displayMode.height != 0) {
        supportedDisplayModes.push_back(displayMode);
      }
    }
    std::sort(supportedDisplayModes.begin(), supportedDisplayModes.end());
    return supportedDisplayModes;
  } else {
    return std::vector<DisplayMode>();
  }
}

std::vector<PixelFormat> Ntv2Discovery::supportedPixelFormat(const Plugin::DiscoveryDevice& device) {
  auto it = std::find_if(
      m_devices.begin(), m_devices.end(),
      [&device](const std::shared_ptr<Device>& m_device) -> bool { return device == m_device->pluginDevice; });
  if (it != m_devices.end()) {
    std::vector<PixelFormat> pixelFormats;
    PixelFormat vsPF;
    // iterate on enum with defined values following themselves : check ntv2enums.h
    for (uint32_t i = NTV2_FBF_10BIT_YCBCR; i < NTV2_FBF_NUMFRAMEBUFFERFORMATS; ++i) {
      if (NTV2DeviceCanDoFrameBufferFormat((*it)->boardInfo.deviceID, (NTV2FrameBufferFormat)i)) {
        // convertPixelFormat((NTV2FrameBufferFormat)i, vsPF);
        vsPF = aja2vsPixelFormat((NTV2FrameBufferFormat)i);
        if (vsPF != Unknown) pixelFormats.push_back(vsPF);
      }
    }
    return pixelFormats;
  } else {
    return std::vector<PixelFormat>();
  }
}

std::vector<int> Ntv2Discovery::supportedNbChannels(const Plugin::DiscoveryDevice& /*device*/) {
  std::vector<int> channels;
  CNTV2DeviceScanner ajaDeviceScanner;
  ajaDeviceScanner.ScanHardware();

  for (uint32_t iDevice = 0; iDevice < ajaDeviceScanner.GetNumDevices(); ++iDevice) {
    channels.push_back((int)ajaDeviceScanner.GetDeviceInfoList()[iDevice].numAudioStreams);
  }

  return channels;
}

std::vector<Audio::SamplingRate> Ntv2Discovery::supportedSamplingRates(const Plugin::DiscoveryDevice& /*device*/) {
  std::vector<Audio::SamplingRate> rates;
  CNTV2DeviceScanner ajaDeviceScanner;
  ajaDeviceScanner.ScanHardware();

  for (uint32_t iDevice = 0; iDevice < ajaDeviceScanner.GetNumDevices(); ++iDevice) {
    NTV2AudioSampleRateList audioSampleRateList = ajaDeviceScanner.GetDeviceInfoList()[iDevice].audioSampleRateList;
    for (auto it = audioSampleRateList.begin(); it != audioSampleRateList.end(); it++) {
      Audio::SamplingRate sRate = convertSamplerate(*it);
      if (sRate != Audio::SamplingRate::SR_NONE) rates.push_back(sRate);
    }
  }

  return rates;
}

std::vector<Audio::SamplingDepth> Ntv2Discovery::supportedSampleFormats(const Plugin::DiscoveryDevice& /*device*/) {
  std::vector<Audio::SamplingDepth> formats;
  CNTV2DeviceScanner ajaDeviceScanner;
  ajaDeviceScanner.ScanHardware();

  for (uint32_t iDevice = 0; iDevice < ajaDeviceScanner.GetNumDevices(); ++iDevice) {
    NTV2AudioBitsPerSampleList audioBitsPerSampleList =
        ajaDeviceScanner.GetDeviceInfoList()[iDevice].audioBitsPerSampleList;
    for (auto it = audioBitsPerSampleList.begin(); it != audioBitsPerSampleList.end(); it++) {
      Audio::SamplingDepth sDepth = convertFormats(*it);
      if (sDepth != Audio::SamplingDepth::SD_NONE) formats.push_back(sDepth);
    }
  }

  return formats;
}

Audio::SamplingRate Ntv2Discovery::convertSamplerate(AudioSampleRateEnum ntv2SampleRate) {
  switch (ntv2SampleRate) {
    case k44p1KHzSampleRate:
      return Audio::SamplingRate::SR_44100;
    case k48KHzSampleRate:
      return Audio::SamplingRate::SR_48000;
    case k96KHzSampleRate:
      return Audio::SamplingRate::SR_96000;
    default:
      return Audio::SamplingRate::SR_NONE;
  }
}

Audio::SamplingDepth Ntv2Discovery::convertFormats(AudioBitsPerSampleEnum ntv2Format) {
  switch (ntv2Format) {
    case k16bitsPerSample:
      return Audio::SamplingDepth::INT16;
    case k24bitsPerSample:
      return Audio::SamplingDepth::INT32;  // 24bits audio PCM are stored in 32bits
      break;
    case k32bitsPerSample:
      return Audio::SamplingDepth::INT32;
    default:
      return Audio::SamplingDepth::SD_NONE;
  }
}
