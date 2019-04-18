// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "portAudioDiscovery.hpp"
#include "libvideostitch/logging.hpp"
#include "portaudio.h"
#include <vector>

namespace VideoStitch {
namespace Plugin {

// ------------------------ Lifecycle --------------------------------
static Audio::SamplingRate paToVsSamplingRate(const double rate) { return Audio::getSamplingRateFromInt(int(rate)); }

static Audio::SamplingDepth paToVsSampleFormat(const PaSampleFormat format) {
  switch (format) {
    case paFloat32:
      return Audio::SamplingDepth::FLT;
    case paInt32:
      return Audio::SamplingDepth::INT32;
    case paInt24:
      return Audio::SamplingDepth::INT24;
    case paInt16:
      return Audio::SamplingDepth::INT16;
    case paUInt8:
      return Audio::SamplingDepth::UINT8;
    case paFloat32 | paNonInterleaved:
      return Audio::SamplingDepth::FLT_P;
    case paInt32 | paNonInterleaved:
      return Audio::SamplingDepth::INT32_P;
    case paInt24 | paNonInterleaved:
      return Audio::SamplingDepth::INT24_P;
    case paInt16 | paNonInterleaved:
      return Audio::SamplingDepth::INT16_P;
    case paUInt8 | paNonInterleaved:
      return Audio::SamplingDepth::UINT8_P;
    default:
      return Audio::SamplingDepth::SD_NONE;
  }
}

static const std::vector<double> PaSampleRateList = {22050.0, 32000.0, 44100.0,  48000.0,
                                                     88200.0, 96000.0, 176400.0, 192000.0};

static const std::vector<PaSampleFormat> PaSampleFormatList = {paFloat32,
                                                               paInt32,
                                                               paInt24,
                                                               paInt16,
                                                               paUInt8,
                                                               paFloat32 | paNonInterleaved,
                                                               paInt32 | paNonInterleaved,
                                                               paInt24 | paNonInterleaved,
                                                               paInt16 | paNonInterleaved,
                                                               paUInt8 | paNonInterleaved};

PortAudioDiscovery* PortAudioDiscovery::create() {
  const PaError initErr = Pa_Initialize();
  if (initErr != paNoError) {
    Logger::get(Logger::Error) << "[PortAudio] Could not initialize PortAudio" << Pa_GetErrorText(initErr) << std::endl;
    return nullptr;
  }

  const int numDevices = Pa_GetDeviceCount();
  if (0 == numDevices) {
    Logger::get(Logger::Error) << "[PortAudio] No devices found" << std::endl;
    Pa_Terminate();
    return nullptr;
  }

  PortAudioDiscovery* devs = new PortAudioDiscovery();

  for (int i = 0; i < numDevices; ++i) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(i);  // Info is not allocated. Do not free
    if (strlen(info->name) == 0) {
      continue;
    }

    std::string apiName(Pa_GetHostApiInfo(info->hostApi) != nullptr ? Pa_GetHostApiInfo(info->hostApi)->name : "");
    std::string newName(info->name);
    newName = newName + " " + apiName;

    // TODO: (Lucas M)
    // For the moment, we support stereo audio only
    if (info->maxInputChannels == 1) {
      continue;
    }
    std::vector<int> supportedChannelCounts;
    for (int nbChannels = 1; nbChannels <= info->maxInputChannels; ++nbChannels) {
      supportedChannelCounts.push_back(nbChannels);
    }
    if (supportedChannelCounts.empty()) {
      continue;
    }

    PaStreamParameters params;
    params.channelCount = info->maxInputChannels;
    params.hostApiSpecificStreamInfo = nullptr;
    params.suggestedLatency = info->defaultLowInputLatency;
    params.device = i;

    // Check for supported sampling formats
    std::vector<Audio::SamplingDepth> supportedFormats;
    PaSampleFormat knownFormat = 0;
    for (auto format : PaSampleFormatList) {
      params.sampleFormat = format;
      if (paFormatIsSupported == Pa_IsFormatSupported(&params, nullptr, info->defaultSampleRate)) {
        supportedFormats.push_back(paToVsSampleFormat(format));
        if (knownFormat == 0) {
          knownFormat = format;
        }
      }
    }
    if (knownFormat == 0) {
      Logger::get(Logger::Warning) << "[PortAudio] Device \"" << info->name
                                   << "\" does not support any know sample formats. Skipping." << std::endl;
      continue;
    }

    // Check for supported sampling rates
    std::vector<Audio::SamplingRate> supportedRates;
    params.sampleFormat = knownFormat;
    supportedRates.push_back(Audio::getSamplingRateFromInt(static_cast<int>(info->defaultSampleRate)));
    for (auto rate : PaSampleRateList) {
      if (rate == info->defaultSampleRate) {
        continue;
      }
      if (paFormatIsSupported == Pa_IsFormatSupported(&params, nullptr, rate)) {
        supportedRates.push_back(paToVsSamplingRate(rate));
      }
    }
    if (supportedRates.empty()) {
      Logger::get(Logger::Warning) << "[PortAudio] Device \"" << info->name
                                   << "\" does not support any know sample rates. Skipping." << std::endl;
      continue;
    }

    devs->_devNames.push_back(newName);
    devs->_devChannelCounts[newName] = supportedChannelCounts;
    devs->_devSampleDepths[newName] = supportedFormats;
    devs->_devSampleRates[newName] = supportedRates;
    devs->_devMaxInputsChannel[newName] = info->maxInputChannels;
    devs->_devMaxOutputsChannel[newName] = info->maxOutputChannels;
  }

  const PaError terminateErr = Pa_Terminate();
  if (terminateErr != paNoError) {
    Logger::get(Logger::Error) << "[PortAudio] Could not terminate PortAudio" << Pa_GetErrorText(terminateErr)
                               << std::endl;
  }

  if (devs->_devNames.empty()) {
    Logger::get(Logger::Warning) << "[PortAudio] No valid devices found" << std::endl;
    delete devs;
    return nullptr;
  }

  return devs;
}

PortAudioDiscovery::PortAudioDiscovery() {}

// -------------------------- Plugin implementation ----------------------------------

void PortAudioDiscovery::registerAutoDetectionCallback(AutoDetection&) {}

std::vector<DisplayMode> PortAudioDiscovery::supportedDisplayModes(const Plugin::DiscoveryDevice&) {
  return std::vector<DisplayMode>();
}

DisplayMode PortAudioDiscovery::currentDisplayMode(const Plugin::DiscoveryDevice&) { return DisplayMode(); }

std::vector<PixelFormat> PortAudioDiscovery::supportedPixelFormat(const Plugin::DiscoveryDevice&) {
  return std::vector<PixelFormat>();
}

std::string PortAudioDiscovery::name() const { return "portaudio"; }

std::string PortAudioDiscovery::readableName() const { return "PortAudio - Portable Cross-platform Audio I/O"; }

std::vector<Plugin::DiscoveryDevice> PortAudioDiscovery::inputDevices() {
  std::vector<Plugin::DiscoveryDevice> devices;
  for (size_t i = 0; i < _devNames.size(); ++i) {
    if (_devMaxInputsChannel[_devNames[i]]) {
      Plugin::DiscoveryDevice device;
      device.name = _devNames[i];
      device.displayName = _devNames[i];
      device.type = Plugin::DiscoveryDevice::CAPTURE;
      device.mediaType = Plugin::DiscoveryDevice::MediaType::AUDIO;
      devices.push_back(device);
    }
  }
  return devices;
}

std::vector<Plugin::DiscoveryDevice> PortAudioDiscovery::outputDevices() {
  std::vector<Plugin::DiscoveryDevice> devices;
  for (size_t i = 0; i < _devNames.size(); ++i) {
    if (_devMaxOutputsChannel[_devNames[i]]) {
      Plugin::DiscoveryDevice device;
      device.name = _devNames[i];
      device.displayName = _devNames[i];
      device.type = Plugin::DiscoveryDevice::PLAYBACK;
      device.mediaType = Plugin::DiscoveryDevice::MediaType::AUDIO;
      devices.push_back(device);
    }
  }
  return devices;
}

std::vector<std::string> PortAudioDiscovery::cards() const { return std::vector<std::string>(); }

std::vector<int> PortAudioDiscovery::supportedNbChannels(const Plugin::DiscoveryDevice& device) {
  if (_devChannelCounts.count(device.name) == 0) {
    return std::vector<int>();
  }
  return _devChannelCounts[device.name];
}

std::vector<Audio::SamplingRate> PortAudioDiscovery::supportedSamplingRates(const Plugin::DiscoveryDevice& device) {
  if (_devSampleRates.count(device.name) == 0) {
    return std::vector<Audio::SamplingRate>();
  }
  return _devSampleRates[device.name];
}

std::vector<Audio::SamplingDepth> PortAudioDiscovery::supportedSampleFormats(const Plugin::DiscoveryDevice& device) {
  if (_devSampleDepths.count(device.name) == 0) {
    return std::vector<Audio::SamplingDepth>();
  }
  return _devSampleDepths[device.name];
}

}  // namespace Plugin
}  // namespace VideoStitch
