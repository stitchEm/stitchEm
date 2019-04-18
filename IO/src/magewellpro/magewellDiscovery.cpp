// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "magewellDiscovery.hpp"
#include "magewell_helpers.hpp"

#include "libvideostitch/logging.hpp"

#include <thread>
#include <future>
#include <chrono>
#include <algorithm>
#include <locale>
#include <codecvt>
#include <unordered_map>
#include <sstream>

using namespace VideoStitch;
using namespace Plugin;

namespace {
// Magewell offers no function to enumerate the video modes available,
// so we're going to test a whitelist of them manually...
// Anyway it's best to use the same mode as the input signal.
std::vector<DisplayMode> getDisplayModesFor(const MWCAP_CHANNEL_INFO& chanInfo) {
  bool is4K = chanInfo.wFamilyID == MW_FAMILY_ID_PRO_CAPTURE &&
              (chanInfo.wProductID == MWCAP_PRODUCT_ID_PRO_CAPTURE_HDMI_4K ||
               chanInfo.wProductID == MWCAP_PRODUCT_ID_PRO_CAPTURE_AIO_4K_PLUS ||
               chanInfo.wProductID == MWCAP_PRODUCT_ID_PRO_CAPTURE_HDMI_4K_PLUS ||
               chanInfo.wProductID == MWCAP_PRODUCT_ID_PRO_CAPTURE_DVI_4K ||
               chanInfo.wProductID == MWCAP_PRODUCT_ID_PRO_CAPTURE_AIO_4K);
  std::vector<DisplayMode> commonDisplayModes;                                 // From DeckLink used formats
  commonDisplayModes.push_back(DisplayMode(720, 486, true, {30000, 1001}));    // NTSC
  commonDisplayModes.push_back(DisplayMode(720, 486, false, {60000, 1001}));   // NTSCp
  commonDisplayModes.push_back(DisplayMode(720, 576, true, {25, 1}));          // PAL
  commonDisplayModes.push_back(DisplayMode(720, 576, false, {50, 1}));         // PALp
  commonDisplayModes.push_back(DisplayMode(1280, 720, false, {50, 1}));        // HD720p50
  commonDisplayModes.push_back(DisplayMode(1280, 720, false, {60000, 1001}));  // HD720p5994
  commonDisplayModes.push_back(DisplayMode(1280, 720, false, {60, 1}));        // HD720p60
  commonDisplayModes.push_back(DisplayMode(1280, 960, false, {25, 1}));
  commonDisplayModes.push_back(DisplayMode(1280, 960, false, {30, 1}));
  commonDisplayModes.push_back(DisplayMode(1280, 960, false, {60, 1}));
  commonDisplayModes.push_back(DisplayMode(1920, 1080, false, {24000, 1001}));  // HD1080p2398
  commonDisplayModes.push_back(DisplayMode(1920, 1080, false, {24, 1}));        // HD1080p24
  commonDisplayModes.push_back(DisplayMode(1920, 1080, false, {25, 1}));        // HD1080p25
  commonDisplayModes.push_back(DisplayMode(1920, 1080, false, {30000, 1001}));  // HD1080p2997
  commonDisplayModes.push_back(DisplayMode(1920, 1080, false, {30, 1}));        // HD1080p30
  commonDisplayModes.push_back(DisplayMode(1920, 1080, true, {25, 1}));         // HD1080i50
  commonDisplayModes.push_back(DisplayMode(1920, 1080, true, {30000, 1001}));   // HD1080i5994
  commonDisplayModes.push_back(DisplayMode(1920, 1080, true, {30, 1}));         // HD1080i60
  commonDisplayModes.push_back(DisplayMode(1920, 1080, false, {48, 1}));
  commonDisplayModes.push_back(DisplayMode(1920, 1080, false, {50, 1}));        // HD1080p50
  commonDisplayModes.push_back(DisplayMode(1920, 1080, false, {60000, 1001}));  // HD1080p5994
  commonDisplayModes.push_back(DisplayMode(1920, 1080, false, {60, 1}));        // HD1080p60
  commonDisplayModes.push_back(DisplayMode(2048, 1556, false, {24000, 1001}));  // 2k1556p2398
  commonDisplayModes.push_back(DisplayMode(2048, 1556, false, {24, 1}));        // 2k1556p24
  commonDisplayModes.push_back(DisplayMode(2048, 1556, false, {25, 1}));        // 2k1556p25
  commonDisplayModes.push_back(DisplayMode(2048, 1080, false, {24000, 1001}));  // 2kDCI2398
  commonDisplayModes.push_back(DisplayMode(2048, 1080, false, {24, 1}));        // 2kDCI24
  commonDisplayModes.push_back(DisplayMode(2048, 1080, false, {25, 1}));        // 2kDCI25
  if (is4K) {
    commonDisplayModes.push_back(DisplayMode(3840, 2160, false, {24000, 1001}));  // 4k2160p2398
    commonDisplayModes.push_back(DisplayMode(3840, 2160, false, {24, 1}));        // 4k2160p24
    commonDisplayModes.push_back(DisplayMode(3840, 2160, false, {25, 1}));        // 4k2160p25
    commonDisplayModes.push_back(DisplayMode(3840, 2160, false, {30000, 1001}));  // 4k2160p2997
    commonDisplayModes.push_back(DisplayMode(3840, 2160, false, {30, 1}));        // 4k2160p30
    commonDisplayModes.push_back(DisplayMode(4096, 2160, false, {24000, 1001}));  // 4kDCI2398
    commonDisplayModes.push_back(DisplayMode(4096, 2160, false, {24, 1}));        // 4kDCI24
    commonDisplayModes.push_back(DisplayMode(4096, 2160, false, {25, 1}));        // 4kDCI25
  }
  return commonDisplayModes;
}
}  // namespace

MagewellDiscovery* MagewellDiscovery::create() {
  BYTE byMaj, byMin;
  WORD wBuild;
  MWGetVersion(&byMaj, &byMin, &wBuild);
  Logger::get(Logger::Info) << "LibMWCapture Version V" << byMaj << "." << byMin << "." << wBuild << std::endl;
  return new MagewellDiscovery;
}

MagewellDiscovery::MagewellDiscovery() : devices() { initializeDevices(); }

MagewellDiscovery::~MagewellDiscovery() {
  for (Device& device : devices) {
    device.listen = false;
  }
  for (std::thread* t : listeners) {
    t->join();
    delete t;
  }
}

void MagewellDiscovery::initializeDevices() {
  assert(m_cards.empty());
  assert(devices.empty());
  std::unordered_map<WORD, std::string> cardsById;
  MWRefreshDevice();

  for (int chan = 0; chan < MWGetChannelCount(); ++chan) {
    MWCAP_CHANNEL_INFO chanInfo = {0};
    if (MW_SUCCEEDED != MWGetChannelInfoByIndex(chan, &chanInfo)) {
      Logger::get(Logger::Error) << "Magewell : Can't get channel info for channel " << chan << std::endl;
      return;
    }

    cardsById[chanInfo.wProductID] = chanInfo.szProductName;

    std::stringstream ss;
    ss << chanInfo.szProductName << " : board " << (int)chanInfo.byBoardIndex << ", channel "
       << (int)chanInfo.byChannelIndex;

    Device device;
    device.pluginDevice.name = std::to_string(chan);
    device.pluginDevice.mediaType = DiscoveryDevice::MediaType::VIDEO;
    device.pluginDevice.type = DiscoveryDevice::CAPTURE;
    device.pluginDevice.displayName = ss.str();
    device.supportedDisplayModes = getDisplayModesFor(chanInfo);
    devices.push_back(device);
  }

  for (const auto& pair : cardsById) {
    m_cards.push_back(pair.second);
  }
}

std::string MagewellDiscovery::name() const { return "magewellpro"; }

std::string MagewellDiscovery::readableName() const { return "Magewell Pro"; }

std::vector<Plugin::DiscoveryDevice> MagewellDiscovery::inputDevices() {
  std::vector<Plugin::DiscoveryDevice> pluginDevices;
  for (const auto& device : devices) {
    pluginDevices.push_back(device.pluginDevice);
  }
  return pluginDevices;
}

std::vector<Plugin::DiscoveryDevice> MagewellDiscovery::outputDevices() {
  return std::vector<Plugin::DiscoveryDevice>();
}

std::vector<std::string> MagewellDiscovery::cards() const { return m_cards; }

void MagewellDiscovery::registerAutoDetectionCallback(AutoDetection& autoDetection) {
  for (Device& device : devices) {
    device.autodetection = &autoDetection;

    if (MWStartVideoCapture(device.channel, device.captureEvent)) {
      Logger::get(Logger::Error) << "Magewell : Open Video Capture error" << std::endl;
      continue;
    }

    std::thread* listener = new std::thread(&MagewellDiscovery::signalDetection, device);
    listeners.push_back(listener);
  }
}

void MagewellDiscovery::signalDetection(Device& device) {
  while (!device.listen) {
    MWCAP_VIDEO_SIGNAL_STATUS videoSignalStatus;
    MWGetVideoSignalStatus(device.channel, &videoSignalStatus);
    if (device.videoSignalState != videoSignalStatus.state) {
      device.videoSignalState = videoSignalStatus.state;
      switch (videoSignalStatus.state) {
        case MWCAP_VIDEO_SIGNAL_NONE:
          Logger::get(Logger::Info) << "Magewell : Input signal status: NONE" << std::endl;
          device.autodetection->signalLost(device.pluginDevice);
          break;
        case MWCAP_VIDEO_SIGNAL_UNSUPPORTED:
          Logger::get(Logger::Info) << "Magewell : Input signal status: Unsupported" << std::endl;
          break;
        case MWCAP_VIDEO_SIGNAL_LOCKING:
          Logger::get(Logger::Info) << "Magewell : Input signal status: Locking" << std::endl;
          break;
        case MWCAP_VIDEO_SIGNAL_LOCKED:
          Logger::get(Logger::Info) << "Magewell : Input signal status: Locked" << std::endl;
          Logger::get(Logger::Info) << "Magewell : Input signal resolution: " << videoSignalStatus.cx << " x "
                                    << videoSignalStatus.cy << std::endl;
          double fps = (double)10000000LL / videoSignalStatus.dwFrameDuration;
          // frameDuration is in nanoseconds
          FrameRate fr;
          if (videoSignalStatus.dwFrameDuration > 416875) {
            fr = {24000, 1001};
          } else if (videoSignalStatus.dwFrameDuration > 408333) {
            fr = {24, 1};
          } else if (videoSignalStatus.dwFrameDuration > 366833) {
            fr = {25, 1};
          } else if (videoSignalStatus.dwFrameDuration > 333500) {
            fr = {30000, 1001};
          } else if (videoSignalStatus.dwFrameDuration > 266667) {
            fr = {30, 1};
          } else if (videoSignalStatus.dwFrameDuration > 183416) {
            fr = {50, 1};
          } else if (videoSignalStatus.dwFrameDuration > 166750) {
            fr = {60000, 1001};
          } else {
            fr = {60, 1};
          }
          Logger::get(Logger::Info) << "Magewell : Input signal fps: " << fps << std::endl;
          Logger::get(Logger::Info) << "Magewell : Input signal interlaced: "
                                    << (videoSignalStatus.bInterlaced ? "true" : "false") << std::endl;
          DisplayMode displayMode = DisplayMode((int64_t)videoSignalStatus.cx, (int64_t)videoSignalStatus.cy,
                                                !(videoSignalStatus.bInterlaced == 0), fr);
          device.autodetection->signalDetected(device.pluginDevice, displayMode);
          break;
      }
    }

    std::this_thread::sleep_for(std::chrono::seconds(3));
  }

  MWStopVideoCapture(device.channel);
  MWCloseChannel(device.channel);
}

std::vector<DisplayMode> MagewellDiscovery::supportedDisplayModes(const Plugin::DiscoveryDevice& device) {
  auto it = std::find_if(devices.cbegin(), devices.cend(),
                         [&device](const Device& d) -> bool { return device == d.pluginDevice; });
  assert(it != devices.cend());
  return it->supportedDisplayModes;
}

std::vector<PixelFormat> MagewellDiscovery::supportedPixelFormat(const Plugin::DiscoveryDevice& /*device*/) {
  std::vector<PixelFormat> pixelFormats;
  pixelFormats.push_back(VideoStitch::BGR);
  pixelFormats.push_back(VideoStitch::YUY2);
  return pixelFormats;
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
