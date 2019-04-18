// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef MAGEWELL_HELPERS_HPP_
#define MAGEWELL_HELPERS_HPP_

#include <windows.h>
#include "LibXIStream/XIStream.h"
#include "DeviceDefs.h"
#include "libvideostitch/frame.hpp"
#include <string>

/**
 * Magewell shared helper functions.
 */
namespace VideoStitch {
namespace Magewell {

XI_COLOR_FORMAT xiColorFormat(const PixelFormat& pixelFormat);

enum class SupportedMagewellCaptureFamily {
  UnknownFamily,
  FirstGenerationCaptureFamily,
  ProCaptureFamily,
  UsbCaptureFamily
};

SupportedMagewellCaptureFamily retrieveCaptureFamily(const VIDEO_CAPTURE_INFO_EX& videoCaptureInfo);
std::string extractMeaningfulDeviceId(const std::string& deviceCompleteId, const std::string& hardwareType);
void extractUsbMeaningfulDeviceIdParts(const std::string& deviceCompleteId, std::string& firstIdPart,
                                       std::string& secondIdPart);
AUDIO_CAPTURE_INFO_EX retrieveAudioCaptureInfo(const VIDEO_CAPTURE_INFO_EX& videoCaptureInfo, bool& ok);

}  // namespace Magewell
}  // namespace VideoStitch

#endif  // MAGEWELL_HELPERS_
