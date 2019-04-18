// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "magewell_discovery.hpp"
#include "magewell_helpers.hpp"

#include "frameRateHelpers.hpp"

#include "libvideostitch/logging.hpp"

#include <thread>
#include <future>
#include <chrono>
#include <algorithm>
#include <locale>
#include <codecvt>

using namespace VideoStitch;
using namespace Plugin;

namespace {
std::vector<DisplayMode> commonDisplayModes() {
  std::vector<DisplayMode> commonDisplayModes;    // From DeckLink used formats
  DisplayMode dm(720, 486, true, {30000, 1001});  // NTSC
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(720, 486, false, {60000, 1001});  // NTSCp
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(720, 576, true, {25, 1});  // PAL
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(720, 576, false, {50, 1});  // PALp
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1280, 720, false, {50, 1});  // HD720p50
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1280, 720, false, {60000, 1001});  // HD720p5994
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1280, 720, false, {60, 1});  // HD720p60
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1280, 960, false, {25, 1});
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1280, 960, false, {30, 1});
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1280, 960, false, {60, 1});
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1920, 1080, false, {24000, 1001});  // HD1080p2398
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1920, 1080, false, {24, 1});  // HD1080p24
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1920, 1080, false, {25, 1});  // HD1080p25
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1920, 1080, false, {30000, 1001});  // HD1080p2997
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1920, 1080, false, {30, 1});  // HD1080p30
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1920, 1080, true, {25, 1});  // HD1080i50
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1920, 1080, true, {30000, 1001});  // HD1080i5994
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1920, 1080, true, {30, 1});  // HD1080i60
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1920, 1080, false, {48, 1});
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1920, 1080, false, {50, 1});  // HD1080p50
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1920, 1080, false, {60000, 1001});  // HD1080p5994
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(1920, 1080, false, {60, 1});  // HD1080p60
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(2048, 1556, false, {24000, 1001});  // 2k1556p2398
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(2048, 1556, false, {24, 1});  // 2k1556p24
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(2048, 1556, false, {25, 1});  // 2k1556p25
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(2048, 1080, false, {24000, 1001});  // 2kDCI2398
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(2048, 1080, false, {24, 1});  // 2kDCI24
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(2048, 1080, false, {25, 1});  // 2kDCI25
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(3840, 2160, false, {24000, 1001});  // 4k2160p2398
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(3840, 2160, false, {24, 1});  // 4k2160p24
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(3840, 2160, false, {25, 1});  // 4k2160p25
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(3840, 2160, false, {30000, 1001});  // 4k2160p2997
  commonDisplayModes.push_back(dm);
  dm = DisplayMode(3840, 2160, false, {30, 1});  // 4k2160p30
  commonDisplayModes.push_back(dm);
  return commonDisplayModes;
}

std::vector<DisplayMode> usbDisplayModes(const VIDEO_CAPTURE_INFO_EX& videoCaptureInfo) {
  static std::vector<DisplayMode> displayModes;
  static std::vector<DisplayMode> displayModes4K;

  if (displayModes.empty()) {
    // From http://www.magewell.com/usb-capture-hdmi/tech-specs
    // and http://www.magewell.com/usb-capture-hdmi-4k-plus/tech-specs
    std::vector<std::pair<int64_t, int64_t>> resolutions;
    resolutions.push_back(std::make_pair(640, 360));
    resolutions.push_back(std::make_pair(640, 480));
    resolutions.push_back(std::make_pair(720, 480));
    resolutions.push_back(std::make_pair(720, 576));
    resolutions.push_back(std::make_pair(768, 576));
    resolutions.push_back(std::make_pair(800, 600));
    resolutions.push_back(std::make_pair(856, 480));
    resolutions.push_back(std::make_pair(960, 540));
    resolutions.push_back(std::make_pair(1024, 576));
    resolutions.push_back(std::make_pair(1024, 768));
    resolutions.push_back(std::make_pair(1280, 720));
    resolutions.push_back(std::make_pair(1280, 800));
    resolutions.push_back(std::make_pair(1280, 960));
    resolutions.push_back(std::make_pair(1280, 1024));
    resolutions.push_back(std::make_pair(1368, 768));
    resolutions.push_back(std::make_pair(1440, 900));
    resolutions.push_back(std::make_pair(1600, 1200));
    resolutions.push_back(std::make_pair(1680, 1050));
    resolutions.push_back(std::make_pair(1920, 1080));
    resolutions.push_back(std::make_pair(1920, 1200));
    std::vector<std::pair<int64_t, int64_t>> resolutions4K;
    resolutions4K.push_back(std::make_pair(2048, 1080));
    resolutions4K.push_back(std::make_pair(2048, 1556));
    resolutions4K.push_back(std::make_pair(3840, 2160));
    resolutions4K.push_back(std::make_pair(4096, 2160));

    std::vector<FrameRate> frameRates;
    frameRates.push_back({25, 1});
    frameRates.push_back({30000, 1001});
    frameRates.push_back({30, 1});
    frameRates.push_back({50, 1});
    frameRates.push_back({60000, 1001});
    frameRates.push_back({60, 1});

    auto nbResolutions = resolutions.size();
    auto nbFrameRates = frameRates.size();
    for (decltype(nbResolutions) resolutionIndex = 0; resolutionIndex < nbResolutions; ++resolutionIndex) {
      for (decltype(nbFrameRates) frameRateIndex = 0; frameRateIndex < nbFrameRates; ++frameRateIndex) {
        displayModes.push_back(DisplayMode(resolutions.at(resolutionIndex).first,
                                           resolutions.at(resolutionIndex).second, false,
                                           frameRates.at(frameRateIndex)));
      }
    }

    displayModes4K = displayModes;
    auto nbResolutions4K = resolutions4K.size();
    for (decltype(nbResolutions4K) resolutionIndex4K = 0; resolutionIndex4K < nbResolutions4K; ++resolutionIndex4K) {
      for (decltype(nbFrameRates) frameRateIndex = 0; frameRateIndex < nbFrameRates; ++frameRateIndex) {
        displayModes4K.push_back(DisplayMode(resolutions4K.at(resolutionIndex4K).first,
                                             resolutions4K.at(resolutionIndex4K).second, false,
                                             frameRates.at(frameRateIndex)));
      }
    }
  }

  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  std::string name = converter.to_bytes(videoCaptureInfo.szName);
  bool is4K = name.find("4K") != std::string::npos;
  return is4K ? displayModes4K : displayModes;
}
}  // namespace

MagewellDiscovery* MagewellDiscovery::create() {
  if (!XIS_Initialize()) {
    Logger::get(Logger::Error)
        << "Error! Magewell library (XIS) could not initialize for discovering devices. Aborting." << std::endl;
    return nullptr;
  }
  return new MagewellDiscovery;
}

MagewellDiscovery::MagewellDiscovery() : m_devices() { initializeDevices(); }

MagewellDiscovery::~MagewellDiscovery() {
  for (Device& device : m_devices) {
    XIP_ClosePropertyHandle(device.hVideoProperty);
    XIS_CloseVideoCapture(device.hVideoCapture);
  }
  XIS_Uninitialize();
}

void VideoStitch::Plugin::MagewellDiscovery::initializeDevices() {
  assert(m_cards.empty());
  assert(m_devices.empty());
  if (!XIS_RefreshDevices()) {
    Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not refresh devices for discovering. Aborting."
                               << std::endl;
    return;
  }

  std::vector<std::string> channelCards;
  for (int iDevice = 0; iDevice < XIS_GetVideoCaptureCount(); ++iDevice) {
    Device device;
    device.pluginDevice.name = std::to_string(iDevice);
    device.pluginDevice.mediaType = DiscoveryDevice::MediaType::VIDEO;
    device.pluginDevice.type = DiscoveryDevice::CAPTURE;

    if (XIS_GetVideoCaptureInfoEx(iDevice, &device.videoCaptureInfo) != TRUE) {
      Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not retrieve informations from the device "
                                 << device.pluginDevice.name << " for discovering." << std::endl;
      continue;
    }
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> wstringToString;
    device.pluginDevice.displayName = wstringToString.to_bytes(device.videoCaptureInfo.szName);

    Magewell::SupportedMagewellCaptureFamily family = Magewell::retrieveCaptureFamily(device.videoCaptureInfo);
    if (family != Magewell::SupportedMagewellCaptureFamily::ProCaptureFamily) {
      m_devices.push_back(device);

      if (family == Magewell::SupportedMagewellCaptureFamily::FirstGenerationCaptureFamily) {
        // We use the audio devide name because only this one enables us to distinguish between 2 cards of the same
        // model
        bool ok = false;
        AUDIO_CAPTURE_INFO_EX audioCaptureInfo = Magewell::retrieveAudioCaptureInfo(device.videoCaptureInfo, ok);
        if (ok) {
          std::string audioDeviceName = wstringToString.to_bytes(std::wstring(audioCaptureInfo.szName));
          auto firstPosition = audioDeviceName.find('(');
          auto lastPosition = audioDeviceName.find(')');
          if (firstPosition != std::string::npos && lastPosition != std::string::npos) {
            channelCards.push_back(audioDeviceName.substr(firstPosition + 1, lastPosition - firstPosition - 1));
          }
        }
      } else {
        m_cards.push_back(device.pluginDevice.displayName);
      }
    }
  }

  // Remove duplicates (because we have the card for each channel)
  std::sort(channelCards.begin(), channelCards.end());
  auto it = std::unique(channelCards.begin(), channelCards.end());
  channelCards.erase(it, channelCards.end());
  m_cards.insert(m_cards.end(), channelCards.begin(), channelCards.end());
}

std::string MagewellDiscovery::name() const { return "magewell"; }

std::string MagewellDiscovery::readableName() const { return "Magewell"; }

std::vector<Plugin::DiscoveryDevice> MagewellDiscovery::inputDevices() {
  std::vector<Plugin::DiscoveryDevice> pluginDevices;
  for (const auto& device : m_devices) {
    pluginDevices.push_back(device.pluginDevice);
  }
  return pluginDevices;
}

std::vector<Plugin::DiscoveryDevice> MagewellDiscovery::outputDevices() {
  return std::vector<Plugin::DiscoveryDevice>();
}

std::vector<std::string> MagewellDiscovery::cards() const { return m_cards; }

void MagewellDiscovery::registerAutoDetectionCallback(AutoDetection& autoDetection) {
  for (Device& device : m_devices) {
    device.autodetection = &autoDetection;

    if (device.hVideoCapture != NULL) {
      XIS_CloseVideoCapture(device.hVideoCapture);
    }
    device.hVideoCapture = XIS_OpenVideoCapture(device.videoCaptureInfo.szDShowID);
    if (device.hVideoCapture == NULL) {
      Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not open the device "
                                 << device.pluginDevice.name << " for discovering." << std::endl;
      continue;
    }

    if (device.hVideoProperty != NULL) {
      XIP_ClosePropertyHandle(device.hVideoProperty);
    }
    device.hVideoProperty = XIS_OpenVideoCapturePropertyHandle(device.hVideoCapture);
    if (device.hVideoProperty == NULL) {
      Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not retrieve the properties of the device "
                                 << device.pluginDevice.name << " for discovering." << std::endl;
      XIS_CloseVideoCapture(device.hVideoCapture);
      continue;
    }

    std::async(std::launch::async, &MagewellDiscovery::signalDetection, device);
  }
}

void MagewellDiscovery::signalDetection(Device& device) {
  Logger::get(Logger::Debug) << "Magewell library (XIS) searching for the signal from the device "
                             << device.pluginDevice.name << "." << std::endl;

  if (XIPHD_IsSignalChanged(device.hVideoProperty) == S_OK) {
    if (XIPHD_IsSignalPresent(device.hVideoProperty) == S_OK) {
      Logger::get(Logger::Debug) << "Magewell library (XIS) detects the signal from the device "
                                 << device.pluginDevice.name << "." << std::endl;
      device.autodetection->signalDetected(device.pluginDevice, MagewellDiscovery::currentDisplayMode(device));
    } else {
      Logger::get(Logger::Debug) << "Magewell library (XIS) losts the signal from the device "
                                 << device.pluginDevice.name << "." << std::endl;
      device.autodetection->signalLost(device.pluginDevice);
    }
  }

  std::this_thread::sleep_for(std::chrono::seconds(3));
  std::async(std::launch::async, &MagewellDiscovery::signalDetection, device);
}

DisplayMode MagewellDiscovery::currentDisplayMode(const Device& device) {
  DisplayMode displayMode = DisplayMode();

  int width, height, frameDuration;
  XIPHD_DEINTERLACE_TYPE deinterlace;
  HRESULT r1 = XIPHD_GetSignalFormat(device.hVideoProperty, &width, &height, &frameDuration);
  HRESULT r2 = XIPHD_GetDeinterlaceType(device.hVideoProperty, &deinterlace);
  if (FAILED(r1) || FAILED(r2)) {
    Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not retrieve the display mode of the device "
                               << device.pluginDevice.name << " for discovering." << std::endl;
    return displayMode;
  }

  displayMode.width = (int64_t)width;
  displayMode.height = (int64_t)height;
  displayMode.interleaved = (deinterlace == XIPHD_DEINTERLACE_NONE) ? false : true;
  double fps = (frameDuration != 0) ? (double)(10000000 / frameDuration) : 0.0;
  displayMode.framerate = Util::fpsToNumDen(fps);

  return displayMode;
}

std::vector<DisplayMode> MagewellDiscovery::supportedDisplayModes(const Plugin::DiscoveryDevice& device) {
  std::vector<DisplayMode> supportedDisplayModes;

  auto it = std::find_if(m_devices.begin(), m_devices.end(),
                         [&device](const Device& d) -> bool { return device == d.pluginDevice; });
  if (it != m_devices.end()) {
    VideoStitch::Magewell::SupportedMagewellCaptureFamily family =
        VideoStitch::Magewell::retrieveCaptureFamily(it->videoCaptureInfo);
    const bool isUsb = (family == VideoStitch::Magewell::SupportedMagewellCaptureFamily::UsbCaptureFamily);
    const std::vector<DisplayMode> displayModes = isUsb ? usbDisplayModes(it->videoCaptureInfo) : commonDisplayModes();
    for (const DisplayMode& displayMode : displayModes) {
      const double fps = double(displayMode.framerate.num) / double(displayMode.framerate.den);
      // XIS_TestVideoCaptureFormat returns always true :(
      if (SUCCEEDED(XIS_TestVideoCaptureFormat(it->hVideoCapture, XI_COLOR_YUYV, (int)displayMode.width,
                                               (int)displayMode.height, (int)(10000000 / fps)))) {
        supportedDisplayModes.push_back(displayMode);
      }
    }
  }

  return supportedDisplayModes;
}

std::vector<PixelFormat> MagewellDiscovery::supportedPixelFormat(const Plugin::DiscoveryDevice& /*device*/) {
  return std::vector<PixelFormat>();
}

std::vector<int> MagewellDiscovery::supportedNbChannels(const Plugin::DiscoveryDevice& /*device*/) {
  std::vector<int> channels;
  channels.push_back(2);
  return channels;
}

std::vector<Audio::SamplingRate> MagewellDiscovery::supportedSamplingRates(const Plugin::DiscoveryDevice& /*device*/) {
  std::vector<Audio::SamplingRate> rates;
  rates.push_back(Audio::SamplingRate::SR_48000);
  return rates;
}

std::vector<Audio::SamplingDepth> MagewellDiscovery::supportedSampleFormats(const Plugin::DiscoveryDevice& /*device*/) {
  std::vector<Audio::SamplingDepth> formats;
  formats.push_back(Audio::SamplingDepth::INT16);
  return formats;
}
