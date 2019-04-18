// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "decklink_helpers.hpp"

#include "libvideostitch/logging.hpp"

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <math.h>

namespace VideoStitch {
namespace DeckLink {

const Helpers::PixelFormatStruct Helpers::defaultPixelFormat = {bmdFormat8BitYUV, PixelFormat::UYVY};
const double Helpers::defaultBytesPerPixel = 2;
const std::map<BMDVideoConnection, std::string> Helpers::videoConnections = {
    {bmdVideoConnectionSDI, "SDI"},
    {bmdVideoConnectionHDMI, "HDMI"},
    {bmdVideoConnectionOpticalSDI, "Optical SDI"},
    {bmdVideoConnectionComponent, "Component"},
    {bmdVideoConnectionComposite, "Composite"},
    {bmdVideoConnectionSVideo, "S-Video"}};

Helpers& Helpers::getInstance() {
  static Helpers deckLinkHelpers;
  return deckLinkHelpers;
}

Helpers::Helpers() {
  // TODO: construct this list dynamically on discovery step, if not we have to updated from the API doc every time
  Plugin::DisplayMode dm(720, 486, true, {30000, 1001});
  m_commonDisplayModes[bmdModeNTSC] = dm;
  dm = Plugin::DisplayMode(720, 486, false, {24000, 1001});
  m_commonDisplayModes[bmdModeNTSC2398] = dm;
  dm = Plugin::DisplayMode(720, 486, false, {60000, 1001});
  m_commonDisplayModes[bmdModeNTSCp] = dm;
  dm = Plugin::DisplayMode(720, 576, true, {25000, 1000});
  m_commonDisplayModes[bmdModePAL] = dm;
  dm = Plugin::DisplayMode(720, 576, false, {50000, 1000});
  m_commonDisplayModes[bmdModePALp] = dm;
  dm = Plugin::DisplayMode(1280, 720, false, {50000, 1000});
  m_commonDisplayModes[bmdModeHD720p50] = dm;
  dm = Plugin::DisplayMode(1280, 720, false, {60000, 1001});
  m_commonDisplayModes[bmdModeHD720p5994] = dm;
  dm = Plugin::DisplayMode(1280, 720, false, {60000, 1000});
  m_commonDisplayModes[bmdModeHD720p60] = dm;
  dm = Plugin::DisplayMode(1920, 1080, false, {24000, 1001});
  m_commonDisplayModes[bmdModeHD1080p2398] = dm;
  dm = Plugin::DisplayMode(1920, 1080, false, {24000, 1000});
  m_commonDisplayModes[bmdModeHD1080p24] = dm;
  dm = Plugin::DisplayMode(1920, 1080, false, {25000, 1000});
  m_commonDisplayModes[bmdModeHD1080p25] = dm;
  dm = Plugin::DisplayMode(1920, 1080, false, {30000, 1001});
  m_commonDisplayModes[bmdModeHD1080p2997] = dm;
  dm = Plugin::DisplayMode(1920, 1080, false, {30000, 1000});
  m_commonDisplayModes[bmdModeHD1080p30] = dm;
  dm = Plugin::DisplayMode(1920, 1080, true, {25000, 1000});
  m_commonDisplayModes[bmdModeHD1080i50] = dm;
  dm = Plugin::DisplayMode(1920, 1080, true, {30000, 1001});
  m_commonDisplayModes[bmdModeHD1080i5994] = dm;
  dm = Plugin::DisplayMode(1920, 1080, true, {30000, 1000});
  m_commonDisplayModes[bmdModeHD1080i6000] = dm;
  dm = Plugin::DisplayMode(1920, 1080, false, {50000, 1000});
  m_commonDisplayModes[bmdModeHD1080p50] = dm;
  dm = Plugin::DisplayMode(1920, 1080, false, {60000, 1001});
  m_commonDisplayModes[bmdModeHD1080p5994] = dm;
  dm = Plugin::DisplayMode(1920, 1080, false, {60000, 1000});
  m_commonDisplayModes[bmdModeHD1080p6000] = dm;
  dm = Plugin::DisplayMode(2048, 1556, false, {24000, 1001});
  m_commonDisplayModes[bmdMode2k2398] = dm;
  dm = Plugin::DisplayMode(2048, 1556, false, {24000, 1000});
  m_commonDisplayModes[bmdMode2k24] = dm;
  dm = Plugin::DisplayMode(2048, 1556, false, {25000, 1000});
  m_commonDisplayModes[bmdMode2k25] = dm;
  dm = Plugin::DisplayMode(2048, 1080, false, {24000, 1001});
  m_commonDisplayModes[bmdMode2kDCI2398] = dm;
  dm = Plugin::DisplayMode(2048, 1080, false, {24000, 1000});
  m_commonDisplayModes[bmdMode2kDCI24] = dm;
  dm = Plugin::DisplayMode(2048, 1080, false, {25000, 1000});
  m_commonDisplayModes[bmdMode2kDCI25] = dm;
  dm = Plugin::DisplayMode(3840, 2160, false, {24000, 1001});
  m_commonDisplayModes[bmdMode4K2160p2398] = dm;
  dm = Plugin::DisplayMode(3840, 2160, false, {24000, 1000});
  m_commonDisplayModes[bmdMode4K2160p24] = dm;
  dm = Plugin::DisplayMode(3840, 2160, false, {25000, 1000});
  m_commonDisplayModes[bmdMode4K2160p25] = dm;
  dm = Plugin::DisplayMode(3840, 2160, false, {30000, 1001});
  m_commonDisplayModes[bmdMode4K2160p2997] = dm;
  dm = Plugin::DisplayMode(3840, 2160, false, {30000, 1000});
  m_commonDisplayModes[bmdMode4K2160p30] = dm;
  dm = Plugin::DisplayMode(4096, 2160, false, {24000, 1001});
  m_commonDisplayModes[bmdMode4kDCI2398] = dm;
  dm = Plugin::DisplayMode(4096, 2160, false, {24000, 1000});
  m_commonDisplayModes[bmdMode4kDCI24] = dm;
  dm = Plugin::DisplayMode(4096, 2160, false, {25000, 1000});
  m_commonDisplayModes[bmdMode4kDCI25] = dm;

  m_commonPixelFormats.push_back({bmdFormat8BitYUV, PixelFormat::UYVY});
  m_commonPixelFormats.push_back({bmdFormat8BitBGRA, PixelFormat::BGRU});
}

BMDDisplayMode Helpers::bmdDisplayMode(const Plugin::DisplayMode& displayMode) const {
  for (const auto& pair : m_commonDisplayModes) {
    if (pair.second == displayMode) {
      return pair.first;
    }
  }
  return bmdModeUnknown;
}

BMDDisplayMode Helpers::bmdDisplayMode(const int64_t width, const int64_t height, const bool interleaved,
                                       const FrameRate framerate) const {
  Plugin::DisplayMode display_mode(width, height, interleaved, framerate);
  display_mode.width = width;
  display_mode.height = height;
  display_mode.interleaved = interleaved;
  display_mode.framerate = framerate;
  return bmdDisplayMode(display_mode);
}

Plugin::DisplayMode Helpers::displayMode(const BMDDisplayMode& bmdDisplayMode) const {
  const auto& pair = m_commonDisplayModes.find(bmdDisplayMode);
  if (pair != m_commonDisplayModes.end()) {
    return pair->second;
  } else {
    return Plugin::DisplayMode(0, 0, false, {1, 1});
  }
}

BMDPixelFormat Helpers::bmdPixelFormat(const PixelFormat& pixelFormat) const {
  const auto& it =
      std::find_if(m_commonPixelFormats.begin(), m_commonPixelFormats.end(),
                   [&pixelFormat](const PixelFormatStruct& pfs) -> bool { return pixelFormat == pfs.pixelFormat; });
  if (it != m_commonPixelFormats.end()) {
    return it->bmdPixelFormat;
  } else {
    return defaultPixelFormat.bmdPixelFormat;
  }
}

BMDPixelFormat Helpers::bmdPixelFormat(const std::string& pixelFormat) const {
  const auto& it = std::find_if(m_commonPixelFormats.begin(), m_commonPixelFormats.end(),
                                [&pixelFormat](const PixelFormatStruct& pfs) -> bool {
                                  return pixelFormat == VideoStitch::getStringFromPixelFormat(pfs.pixelFormat);
                                });
  if (it != m_commonPixelFormats.end()) {
    return it->bmdPixelFormat;
  } else {
    return defaultPixelFormat.bmdPixelFormat;
  }
}

PixelFormat Helpers::pixelFormat(const BMDPixelFormat& bmdPixelFormat) const {
  const auto& it = std::find_if(
      m_commonPixelFormats.begin(), m_commonPixelFormats.end(),
      [&bmdPixelFormat](const PixelFormatStruct& pfs) -> bool { return bmdPixelFormat == pfs.bmdPixelFormat; });
  if (it != m_commonPixelFormats.end()) {
    return it->pixelFormat;
  } else {
    return defaultPixelFormat.pixelFormat;
  }
}

std::string Helpers::pixelFormatName(const PixelFormat& pixelFormat) const {
  const auto& it =
      std::find_if(m_commonPixelFormats.begin(), m_commonPixelFormats.end(),
                   [&pixelFormat](const PixelFormatStruct& pfs) -> bool { return pixelFormat == pfs.pixelFormat; });
  if (it != m_commonPixelFormats.end()) {
    return VideoStitch::getStringFromPixelFormat(it->pixelFormat);
  } else {
    return VideoStitch::getStringFromPixelFormat(defaultPixelFormat.pixelFormat);
  }
}

std::string Helpers::videoConnectionToString(const BMDVideoConnection videoConnection) const {
  std::string videoConnectionsString;
  for (auto _videoConnection : videoConnections) {
    if (videoConnection & _videoConnection.first) {
      if (!videoConnectionsString.empty()) {
        videoConnectionsString += " & ";
      }
      videoConnectionsString += _videoConnection.second;
    }
  }
  return videoConnectionsString;
}

// From the documentation 2.7.4 Pixel Formats section (start at page 295)
int64_t frameSize(const BMDPixelFormat& format, int64_t width, int64_t height) {
  switch (format) {
    case bmdFormat8BitYUV:
      return (width * 16 / 8) * height;
    case bmdFormat10BitYUV:
      return int64_t(((float(width) + 47.f) / 48.f) * 128.f * float(height));
    case bmdFormat8BitBGRA:
    case bmdFormat8BitARGB:
      return (width * 32 / 8) * height;
    case bmdFormat10BitRGB:
    case bmdFormat10BitRGBXLE:
    case bmdFormat10BitRGBX:
      return int64_t(((float(width) + 63.f) / 64.f) * 256.f * float(height));
  }
  return 0;
}

// From the documentation 2.7.4 Pixel Formats section (start at page 295)
double Helpers::bytesPerPixel(const BMDPixelFormat& bmdPixelFormat) const {
  switch (bmdPixelFormat) {
    case bmdFormat8BitYUV:
      return 2;
    case bmdFormat10BitYUV:
      return 6 / 16;
    case bmdFormat8BitBGRA:
    case bmdFormat8BitARGB:
    case bmdFormat10BitRGB:
    case bmdFormat10BitRGBXLE:
    case bmdFormat10BitRGBX:
      return 4;
    default:
      return defaultBytesPerPixel;
  }
}

std::string videoModeToString(const Plugin::DisplayMode& displayMode, const PixelFormat& pixelFormat) {
  return displayModeToString(displayMode) + " in " + Helpers::getInstance().pixelFormatName(pixelFormat) +
         " pixel format";
}

std::string displayModeToString(const Plugin::DisplayMode& displayMode) {
  return std::to_string(displayMode.width) + "x" + std::to_string(displayMode.height) +
         ((displayMode.interleaved) ? 'i' : 'p') + " at " + std::to_string(displayMode.framerate.num) + "/" +
         std::to_string(displayMode.framerate.den) + " fps";
}

std::string fieldDominanceToString(const BMDFieldDominance& fieldDominance) {
  switch (fieldDominance) {
    case bmdLowerFieldFirst:
      return "Lower field first";
    case bmdUpperFieldFirst:
      return "Upper field first";
    case bmdProgressiveFrame:
      return "Progressive";
    case bmdProgressiveSegmentedFrame:
      return "Progressive segmented frame";
    case bmdUnknownFieldDominance:
    default:
      return "Unknown";
  }
}

std::string colotSpaceToString(const BMDDetectedVideoInputFormatFlags& flags) {
  if (flags == bmdDetectedVideoInputYCbCr422) {
    return "YCbCr422";
  } else if (flags == bmdDetectedVideoInputRGB444) {
    return "RGB444";
  } else {
    return "Unknown";
  }
}

double round(double number) { return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5); }

std::shared_ptr<IDeckLinkIterator> createIterator() {
  IDeckLinkIterator* decklinkIterator = nullptr;
#if defined(WIN32)
  CoInitialize(nullptr);
  if (FAILED(CoCreateInstance(CLSID_CDeckLinkIterator, nullptr, CLSCTX_ALL, IID_IDeckLinkIterator,
                              (void**)&decklinkIterator))) {
    Logger::get(Logger::Warning)
        << "Warning! Please install the DeckLink drivers if you want to use this type of cards." << std::endl;
    return std::shared_ptr<IDeckLinkIterator>();
  }
#else
  decklinkIterator = CreateDeckLinkIteratorInstance();
  if (decklinkIterator == nullptr) {
    Logger::get(Logger::Error)
        << "A DeckLink iterator could not be created.  The DeckLink drivers may not be installed." << std::endl;
    return std::shared_ptr<IDeckLinkIterator>();
  }
#endif
  return std::shared_ptr<IDeckLinkIterator>(decklinkIterator, getDefaultDeleter());
}

// Not normalized name, don't use it in the writer / reader / discovery
// Returns <decklink display name, decklink model name>
std::pair<std::string, std::string> retrieveDeviceInfos(std::shared_ptr<IDeckLink> deckLinkDevice) {
  // GetDisplayName returns a name to identify the device
  // GetModelName returns the card name
  std::pair<std::string, std::string> pairOfNames;
#if defined(WIN32)
  BSTR displayNameBSTR, modelNameBSTR;
  if (FAILED(deckLinkDevice->GetDisplayName(&displayNameBSTR)) ||
      FAILED(deckLinkDevice->GetModelName(&modelNameBSTR))) {
    SysFreeString(displayNameBSTR);
    SysFreeString(modelNameBSTR);
    return std::pair<std::string, std::string>();
  }
  pairOfNames = std::make_pair(DeckLink::BSTRtoString(displayNameBSTR), DeckLink::BSTRtoString(modelNameBSTR));
  SysFreeString(displayNameBSTR);
  SysFreeString(modelNameBSTR);
#else
  char* displayNameString = nullptr;
  char* modelNameString = nullptr;
  if (FAILED(deckLinkDevice->GetDisplayName((const char**)&displayNameString)) ||
      FAILED(deckLinkDevice->GetModelName((const char**)&modelNameString))) {
    return std::pair<std::string, std::string>();
  }
  pairOfNames = std::make_pair(std::string((const char*)displayNameString), std::string((const char*)modelNameString));
#endif
  return pairOfNames;
}

std::vector<std::string> retrieveCardsNames() {
  std::vector<std::string> cards;
  std::shared_ptr<IDeckLinkIterator> iterator = VideoStitch::DeckLink::createIterator();
  IDeckLink* tempDevice = nullptr;

  while (iterator && iterator->Next(&tempDevice) == S_OK) {
    std::shared_ptr<IDeckLink> device(tempDevice, getDefaultDeleter());
    auto infos = VideoStitch::DeckLink::retrieveDeviceInfos(device);
    cards.push_back(infos.second);
  }

  // Remove duplicates (because we have the card for each channel)
  std::sort(cards.begin(), cards.end());
  auto it = std::unique(cards.begin(), cards.end());
  cards.erase(it, cards.end());
  return cards;
}

std::vector<std::pair<std::string, std::string>> retrieveDevicesNames() {
  std::shared_ptr<IDeckLinkIterator> iterator = VideoStitch::DeckLink::createIterator();
  IDeckLink* tempDevice = nullptr;
  std::vector<std::pair<std::string, std::string>> names;

  while (iterator && iterator->Next(&tempDevice) == S_OK) {
    // Retrieve decklink objects
    std::shared_ptr<IDeckLink> device(tempDevice, getDefaultDeleter());

    IDeckLinkConfiguration* tempConfiguration = nullptr;
    if (FAILED(device->QueryInterface(IID_IDeckLinkConfiguration, (void**)&tempConfiguration))) {
      Logger::get(Logger::Error) << "Error! DeckLink API could not to obtain the DeckLinkConfiguration interface"
                                 << std::endl;
      continue;
    }
    std::shared_ptr<IDeckLinkConfiguration> configuration(tempConfiguration, getDefaultDeleter());

    IDeckLinkAttributes* tempAttributes = nullptr;
    if (FAILED(device->QueryInterface(IID_IDeckLinkAttributes, (void**)&tempAttributes))) {
      Logger::get(Logger::Error) << "Error! DeckLink API could not to obtain the DeckLinkAttributes interface"
                                 << std::endl;
      continue;
    }
    std::shared_ptr<IDeckLinkAttributes> attributes(tempAttributes, getDefaultDeleter());

    // Retrieve the needed infos
    auto infos = VideoStitch::DeckLink::retrieveDeviceInfos(device);

    // Compute the names
    // We need the model name to distinguish between Quad and Quad 2, we don't know the naming behaviour of Decklink in
    // that case
    std::string idName = infos.second + " - " + infos.first;
    // With Decklink drivers 10.7, they fixed VSA-3586 that we reported to them
    // but they use the same method to distinguish between the cards and to distinguish between the subdevices
    // so we need all these informations in the displayable name
    std::string displayableName = infos.second == infos.first ? infos.second : infos.second + " - " + infos.first;

    int64_t nbSubDevices = 0;
    attributes->GetInt(BMDDeckLinkNumberOfSubDevices, &nbSubDevices);
    if (nbSubDevices >= 2) {
      int64_t subDeviceIndex = 0;
      attributes->GetInt(BMDDeckLinkSubDeviceIndex, &subDeviceIndex);
      int64_t nbPairs = nbSubDevices / 2;
      int64_t connectorId = (subDeviceIndex % nbPairs) * 2 + subDeviceIndex / nbPairs + 1;
      displayableName += std::string(" - Connector ") + std::to_string(connectorId);
    }

#if defined(WIN32)
    LONGLONG videoConnection;
#else
    int64_t videoConnection;
#endif
    if (SUCCEEDED(configuration->GetInt(bmdDeckLinkConfigVideoInputConnection, &videoConnection))) {
      displayableName += std::string(" (") +
                         DeckLink::Helpers::getInstance().videoConnectionToString((BMDVideoConnection)videoConnection) +
                         std::string(")");
    }

    names.push_back(std::make_pair(idName, displayableName));
  }

  return names;
}

std::shared_ptr<IDeckLink> retrieveDevice(const std::string& deviceName) {
  auto names = retrieveDevicesNames();
  auto it = std::find_if(
      names.cbegin(), names.cend(),
      [&deviceName](const std::pair<std::string, std::string>& infos) -> bool { return deviceName == infos.first; });
  if (it == names.cend()) {
    return std::shared_ptr<IDeckLink>();
  }

  // Find the device corresponding to the name
  int indexToFind = int(it - names.cbegin());
  int otherIndex = 0;
  IDeckLink* tempDeckLinkDevice = nullptr;
  std::shared_ptr<IDeckLinkIterator> decklinkIterator = VideoStitch::DeckLink::createIterator();
  while (decklinkIterator->Next(&tempDeckLinkDevice) == S_OK) {
    std::shared_ptr<IDeckLink> deckLinkDevice(tempDeckLinkDevice, getDefaultDeleter());
    if (otherIndex == indexToFind) {
      return deckLinkDevice;
    }
    ++otherIndex;
  }
  return std::shared_ptr<IDeckLink>();
}

std::shared_ptr<IDeckLinkConfiguration> configureDuplexMode(std::shared_ptr<IDeckLink> subDevice) {
  // This method is usefull only for Quad 2 and Duo 2 which support configuration of the duplex mode of individual
  // sub-devices
  std::shared_ptr<IDeckLink> configurableSubDevice = subDevice;

  IDeckLinkAttributes* tempAttributes = nullptr;
  if (subDevice->QueryInterface(IID_IDeckLinkAttributes, (void**)&tempAttributes) != S_OK) {
    Logger::get(Logger::Error) << "Error! DeckLink API could not to obtain the DeckLinkAttributes interface"
                               << std::endl;
    return std::shared_ptr<IDeckLinkConfiguration>();
  }
  std::shared_ptr<IDeckLinkAttributes> attributes(tempAttributes, getDefaultDeleter());

  BOOL supportDuplexMode = false;
  attributes->GetFlag(BMDDeckLinkSupportsDuplexModeConfiguration, &supportDuplexMode);

  // This sub device does not support duplex mode configuration, we will try with the paired sub-device
  if (!supportDuplexMode) {
    LONGLONG otherSubDeviceId = 0;
    if (attributes->GetInt(BMDDeckLinkPairedDevicePersistentID, &otherSubDeviceId) != S_OK || otherSubDeviceId == 0) {
      return std::shared_ptr<IDeckLinkConfiguration>();
    }

    std::shared_ptr<IDeckLinkIterator> it = VideoStitch::DeckLink::createIterator();
    IDeckLink* tempOtherSubDevice = nullptr;
    while (it && it->Next(&tempOtherSubDevice) == S_OK) {
      std::shared_ptr<IDeckLink> otherSubDevice(tempOtherSubDevice, getDefaultDeleter());

      IDeckLinkAttributes* tempOtherAttributes = nullptr;
      if (otherSubDevice->QueryInterface(IID_IDeckLinkAttributes, (void**)&tempOtherAttributes) != S_OK) {
        Logger::get(Logger::Error) << "Error! DeckLink API could not to obtain the DeckLinkAttributes interface"
                                   << std::endl;
        return std::shared_ptr<IDeckLinkConfiguration>();
      }
      std::shared_ptr<IDeckLinkAttributes> otherAttributes(tempOtherAttributes, getDefaultDeleter());

      LONGLONG id = 0;
      otherAttributes->GetInt(BMDDeckLinkPersistentID, &id);
      if (id == otherSubDeviceId) {
        otherAttributes->GetFlag(BMDDeckLinkSupportsDuplexModeConfiguration, &supportDuplexMode);
        configurableSubDevice = otherSubDevice;
        break;
      }
    }
  }

  if (!supportDuplexMode) {
    return std::shared_ptr<IDeckLinkConfiguration>();
  }

  // If the duplex mode is already well configured, we will not configure this again
  // Also this avoid a bug (DeckLinkReader::VideoInputFrameArrived never called) in the case where
  // we have the 2nd sub-device before the 1st sub-device in the vah
  IDeckLinkStatus* tempStatus = nullptr;
  if (subDevice->QueryInterface(IID_IDeckLinkStatus, (void**)&tempStatus) != S_OK) {
    Logger::get(Logger::Error) << "Error! DeckLink API could not to obtain the DeckLinkStatus interface" << std::endl;
    return std::shared_ptr<IDeckLinkConfiguration>();
  }
  LONGLONG duplexStatus = bmdDuplexStatusInactive;
  tempStatus->GetInt(bmdDeckLinkStatusDuplexMode, &duplexStatus);
  tempStatus->Release();

  if (duplexStatus == bmdDuplexStatusHalfDuplex) {
    return std::shared_ptr<IDeckLinkConfiguration>();
  }

  // Now we have the right sub device, we can configure it in half duplex mode
  IDeckLinkConfiguration* tempConfiguration = nullptr;
  if (subDevice->QueryInterface(IID_IDeckLinkConfiguration, (void**)&tempConfiguration) != S_OK) {
    Logger::get(Logger::Error) << "Error! DeckLink API could not to obtain the DeckLinkConfiguration interface"
                               << std::endl;
    return std::shared_ptr<IDeckLinkConfiguration>();
  }

  std::shared_ptr<IDeckLinkConfiguration> configuration(tempConfiguration, getDefaultDeleter());
  configuration->SetInt(bmdDeckLinkConfigDuplexMode, bmdDuplexModeHalf);
  return configuration;
}

std::function<void(IUnknown*)> getDefaultDeleter() {
  return [](IUnknown* p) {
    if (p) {
      p->Release();
    }
  };
}

#if defined(_WIN32)
std::string BSTRtoString(BSTR bstr) {  // Thanks to http://stackoverflow.com/q/6284524
  int wslen = ::SysStringLen(bstr);
  const wchar_t* pstr = static_cast<wchar_t*>(bstr);
  int len = ::WideCharToMultiByte(CP_ACP, 0, pstr, wslen, NULL, 0, NULL, NULL);

  std::string dblstr(len, '\0');
  len = ::WideCharToMultiByte(CP_ACP, 0 /* no flags */, pstr, wslen /* not necessary NULL-terminated */, &dblstr[0],
                              len, NULL, NULL /* no default char */);
  return dblstr;
}

#endif

}  // namespace DeckLink
}  // namespace VideoStitch
