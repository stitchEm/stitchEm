// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/plugin.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>

#if defined(_WIN32)
#include "DeckLinkAPI_h.h"
#else
#include "DeckLinkAPI.h"
#endif

/**
 * DeckLink shared helper functions.
 */
namespace VideoStitch {
namespace DeckLink {

class Helpers {
 public:
  struct PixelFormatStruct {
    const BMDPixelFormat bmdPixelFormat;
    const PixelFormat pixelFormat;
  };

  static Helpers& getInstance();

  const std::map<BMDDisplayMode, Plugin::DisplayMode>& commonDisplayModes() const { return m_commonDisplayModes; }
  const std::vector<PixelFormatStruct>& commonPixelFormats() const { return m_commonPixelFormats; }

  BMDDisplayMode bmdDisplayMode(const Plugin::DisplayMode&) const;
  BMDDisplayMode bmdDisplayMode(const int64_t width, const int64_t height, const bool interleaved,
                                const FrameRate framerate) const;
  Plugin::DisplayMode displayMode(const BMDDisplayMode&) const;

  BMDPixelFormat bmdPixelFormat(const PixelFormat&) const;
  BMDPixelFormat bmdPixelFormat(const std::string&) const;
  PixelFormat pixelFormat(const BMDPixelFormat&) const;
  std::string pixelFormatName(const PixelFormat&) const;

  std::string videoConnectionToString(const BMDVideoConnection videoConnection) const;

  double bytesPerPixel(const BMDPixelFormat&) const;

 private:
  Helpers();
  Helpers(const Helpers&);
  Helpers& operator=(const Helpers&);

  std::map<BMDDisplayMode, Plugin::DisplayMode> m_commonDisplayModes;
  std::vector<PixelFormatStruct> m_commonPixelFormats;

  static const PixelFormatStruct defaultPixelFormat;
  static const double defaultBytesPerPixel;
  static const std::map<BMDVideoConnection, std::string> videoConnections;
  static const double doubleEpsilon;
};

int64_t frameSize(const BMDPixelFormat&, const int64_t width, const int64_t height);
std::string videoModeToString(const Plugin::DisplayMode& displayMode, const PixelFormat& pixelFormat);
std::string displayModeToString(const Plugin::DisplayMode& displayMode);
std::string fieldDominanceToString(const BMDFieldDominance& fieldDominance);
std::string colotSpaceToString(const BMDDetectedVideoInputFormatFlags& flags);
double round(double number);
std::shared_ptr<IDeckLinkIterator> createIterator();
std::vector<std::string> retrieveCardsNames();
std::vector<std::pair<std::string, std::string>>
retrieveDevicesNames();  // Returns a list of <device id name, device displayable name>
std::shared_ptr<IDeckLink> retrieveDevice(const std::string& deviceName);
std::shared_ptr<IDeckLinkConfiguration> configureDuplexMode(std::shared_ptr<IDeckLink> subDevice);
std::function<void(IUnknown*)> getDefaultDeleter();

#if defined(_WIN32)
std::string BSTRtoString(BSTR bstr);
#endif
}  // namespace DeckLink
}  // namespace VideoStitch
