// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "magewell_helpers.hpp"

#include "libvideostitch/logging.hpp"

#include <codecvt>

namespace VideoStitch {
namespace Magewell {

SupportedMagewellCaptureFamily retrieveCaptureFamily(const VIDEO_CAPTURE_INFO_EX& videoCaptureInfo) {
  if (videoCaptureInfo.deviceType == XI_DEVICE_HDVIDEO_CAPTURE) {
    return SupportedMagewellCaptureFamily::FirstGenerationCaptureFamily;
  } else if (videoCaptureInfo.deviceType == PRO_CAPTURE) {
    return SupportedMagewellCaptureFamily::ProCaptureFamily;
  } else {
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    std::string videoDeviceCompleteId = converter.to_bytes(std::wstring(videoCaptureInfo.szDShowID));
    if (videoDeviceCompleteId.find("usb") != std::string::npos) {
      return SupportedMagewellCaptureFamily::UsbCaptureFamily;
    } else {
      return SupportedMagewellCaptureFamily::UnknownFamily;
    }
  }
}

std::string extractMeaningfulDeviceId(const std::string& deviceCompleteId, const std::string& hardwareType) {
  size_t hardwarePosition = deviceCompleteId.find(hardwareType);
  size_t bracePosition = deviceCompleteId.find("{", hardwarePosition);
  if (hardwarePosition == std::string::npos || bracePosition == std::string::npos) {
    return std::string();
  } else {
    return deviceCompleteId.substr(hardwarePosition, bracePosition - hardwarePosition);
  }
}

void extractUsbMeaningfulDeviceIdParts(const std::string& deviceCompleteId, std::string& firstIdPart,
                                       std::string& secondIdPart) {
  std::string cardId = extractMeaningfulDeviceId(deviceCompleteId, "usb");
  if (cardId.empty()) {
    return;
  }

  size_t lastAndPosition = cardId.rfind("&");
  size_t lastSharpPosition = cardId.rfind("#", lastAndPosition);
  secondIdPart = cardId.substr(lastSharpPosition, lastAndPosition - lastSharpPosition);

  size_t previousAndPosition = cardId.rfind("&", lastSharpPosition);
  firstIdPart = cardId.substr(0, previousAndPosition);
}

AUDIO_CAPTURE_INFO_EX retrieveAudioCaptureInfo(const VIDEO_CAPTURE_INFO_EX& videoCaptureInfo, bool& ok) {
  // For Magewell, audio devices and video devices are not in the same order. So we can't use the device index.
  // This method parse the device id to retrieve the audio device from the video device
  // We can also use the code of the example named "DeviceMatch" in the latest version of the Magewell SDK 2
  // (2.1.0.2008)
  ok = false;

  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  std::string videoDeviceCompleteId = converter.to_bytes(std::wstring(videoCaptureInfo.szDShowID));

  SupportedMagewellCaptureFamily family = retrieveCaptureFamily(videoCaptureInfo);
  switch (family) {
    case SupportedMagewellCaptureFamily::FirstGenerationCaptureFamily:
    case SupportedMagewellCaptureFamily::ProCaptureFamily: {
      const char videoDeviceIndex = videoDeviceCompleteId.back();
      std::string videoCardId = extractMeaningfulDeviceId(videoDeviceCompleteId, "pci");
      if (videoCardId.empty()) {
        Logger::get(Logger::Error) << "Error! Magewell video device id (" << videoDeviceCompleteId
                                   << ") not valid. Aborting." << std::endl;
        return AUDIO_CAPTURE_INFO_EX();
      }

      // Stupid Magewell SDK lists all the audio input devices (including the microphone) in random order
      int nbAudioCaptures = XIS_GetAudioCaptureCount();
      for (int index = 0; index < nbAudioCaptures; ++index) {
        AUDIO_CAPTURE_INFO_EX audioCaptureInfo;
        if (XIS_GetAudioCaptureInfoEx(index, &audioCaptureInfo) != TRUE) {
          Logger::get(Logger::Error)
              << "Error! Magewell library (XIS) could not list available audio devices. Aborting." << std::endl;
          return AUDIO_CAPTURE_INFO_EX();
        }

        std::string deviceName = converter.to_bytes(std::wstring(audioCaptureInfo.szName));
        std::string audioDeviceCompleteId = converter.to_bytes(std::wstring(audioCaptureInfo.szID));
        Logger::get(Logger::Debug) << "Magewell library (XIS): found audio capture device name: " << deviceName << "."
                                   << std::endl
                                   << "Device id: " << audioDeviceCompleteId << std::endl;

        const char audioDeviceIndex = audioDeviceCompleteId.back();
        std::string audioCardId = extractMeaningfulDeviceId(audioDeviceCompleteId, "pci");
        if (audioCardId.empty()) {
          // Not a Magewell audio device
          continue;
        }

        if (videoCardId == audioCardId &&
            (videoDeviceIndex == audioDeviceIndex || family == SupportedMagewellCaptureFamily::ProCaptureFamily)) {
          Logger::get(Logger::Debug) << "Magewell library (XIS): found suitable audio capture device." << std::endl;
          ok = true;
          return audioCaptureInfo;
        }
      }
      break;
    }
    case SupportedMagewellCaptureFamily::UsbCaptureFamily: {
      std::string firstVideoIdPart, secondVideoIdPart;
      extractUsbMeaningfulDeviceIdParts(videoDeviceCompleteId, firstVideoIdPart, secondVideoIdPart);
      if (firstVideoIdPart.empty() || secondVideoIdPart.empty()) {
        Logger::get(Logger::Error) << "Error! Magewell video device id (" << videoDeviceCompleteId
                                   << ") not valid. Aborting." << std::endl;
        return AUDIO_CAPTURE_INFO_EX();
      }

      // Stupid Magewell SDK lists all the audio input devices (including the microphone) in random order
      int nbAudioCaptures = XIS_GetAudioCaptureCount();
      for (int index = 0; index < nbAudioCaptures; ++index) {
        AUDIO_CAPTURE_INFO_EX audioCaptureInfo;
        if (XIS_GetAudioCaptureInfoEx(index, &audioCaptureInfo) != TRUE) {
          Logger::get(Logger::Error)
              << "Error! Magewell library (XIS) could not list available audio devices. Aborting." << std::endl;
          return AUDIO_CAPTURE_INFO_EX();
        }

        std::string deviceName = converter.to_bytes(std::wstring(audioCaptureInfo.szName));
        std::string audioDeviceCompleteId = converter.to_bytes(std::wstring(audioCaptureInfo.szID));
        Logger::get(Logger::Debug) << "Magewell library (XIS): found audio capture device name: " << deviceName << "."
                                   << std::endl
                                   << "Device id: " << audioDeviceCompleteId << std::endl;

        std::string firstAudioIdPart, secondAudioIdPart;
        extractUsbMeaningfulDeviceIdParts(audioDeviceCompleteId, firstAudioIdPart, secondAudioIdPart);
        if (firstAudioIdPart.empty() || secondAudioIdPart.empty()) {
          // Not a Magewell audio device
          continue;
        }

        if (firstVideoIdPart == firstAudioIdPart && secondVideoIdPart == secondAudioIdPart) {
          Logger::get(Logger::Debug) << "Magewell library (XIS): found suitable audio capture device." << std::endl;
          ok = true;
          return audioCaptureInfo;
        }
      }
      break;
    }
    case SupportedMagewellCaptureFamily::UnknownFamily:
    default:
      break;  // nothing
  }

  Logger::get(Logger::Error)
      << "Error! Magewell library (XIS): aborting. No suitable audio device found for associated video device id: "
      << videoDeviceCompleteId << std::endl;
  return AUDIO_CAPTURE_INFO_EX();
}

}  // namespace Magewell
}  // namespace VideoStitch
