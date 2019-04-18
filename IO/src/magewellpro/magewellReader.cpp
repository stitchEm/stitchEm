// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "magewellReader.hpp"
#include "magewell_helpers.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Input {

// ------------------------ Lifecycle --------------------------------

MagewellReader::MagewellReader(readerid_t id, const int64_t width, const int64_t height, HCHANNEL channel,
                               int64_t frameSize, FrameRate fps, PixelFormat pixelFormat, int bytesPerPixel,
                               const std::string& name)
    : Reader(id),
      VideoReader(width, height, frameSize, pixelFormat, Host, fps, 0, NO_LAST_FRAME,
                  false /* not a procedural reader */, nullptr),
      channel(channel),
      name(name),
      bytesPerPixel(bytesPerPixel) {
  notifyEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
  captureEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
}

MagewellReader::~MagewellReader() {
  MWStopVideoCapture(channel);
  CloseHandle(notifyEvent);
  CloseHandle(captureEvent);
  MWCloseChannel(channel);
}

// -------------------------- Plugin implementation ----------------------------------

MagewellReader* MagewellReader::create(readerid_t id, const Ptv::Value* config, const int64_t width,
                                       const int64_t height) {
  MWRefreshDevice();

  std::string name;
  if (Parse::populateString("Magewell reader", *config, "name", name, true) != VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error)
        << "Magewell : camera name (\"name\") couldn't be retrieved. Please, give a name to your input. Aborting."
        << std::endl;
    return nullptr;
  }

  //------------------------ Video
  FrameRate frameRate;
  if (!config->has("frame_rate")) {
    Logger::get(Logger::Error) << "Magewell : frame rate (\"frame_rate\") couldn't be retrieved. Aborting."
                               << std::endl;
    return nullptr;
  } else {
    const Ptv::Value* fpsConf = config->has("frame_rate");
    if ((Parse::populateInt("Magewell reader", *fpsConf, "num", frameRate.num, false) !=
         VideoStitch::Parse::PopulateResult_Ok) ||
        (Parse::populateInt("Magewell reader", *fpsConf, "den", frameRate.den, false) !=
         VideoStitch::Parse::PopulateResult_Ok)) {
      Logger::get(Logger::Error) << "Magewell : frame rate (\"frame_rate\") couldn't be retrieved. Aborting."
                                 << std::endl;
      return nullptr;
    }
  }

  std::string pixelFormatString;
  if (Parse::populateString("Magewell reader", *config, "pixel_format", pixelFormatString, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "Magewell : pixel format (\"pixel_format\") couldn't be retrieved. Aborting."
                               << std::endl;
    return nullptr;
  }
  PixelFormat pixelFormat = PixelFormat::Unknown;
  int bytesPerPixel = 0;
  if (pixelFormatString == "RGBA") {
    pixelFormat = PixelFormat::RGBA;
    bytesPerPixel = 4;
  } else if (pixelFormatString == "RGB") {
    pixelFormat = PixelFormat::RGB;
    bytesPerPixel = 3;
  } else if (pixelFormatString == "BGR") {
    pixelFormat = PixelFormat::BGR;
    bytesPerPixel = 3;
  } else if (pixelFormatString == "BGRU") {
    pixelFormat = PixelFormat::BGRU;
    bytesPerPixel = 4;
  } else if (pixelFormatString == "UYVY") {
    pixelFormat = PixelFormat::UYVY;
    bytesPerPixel = 2;
  } else if (pixelFormatString == "YUY2") {
    pixelFormat = PixelFormat::YUY2;
    bytesPerPixel = 2;
  }

  //------------------------------- Runtime objects
  MWCAP_CHANNEL_INFO videoInfo = {0};
  if (MW_SUCCEEDED != MWGetChannelInfoByIndex(std::stoi(name), &videoInfo)) {
    Logger::get(Logger::Error) << "Magewell : Can't get channel info" << std::endl;
    return nullptr;
  }
  HCHANNEL channel = MWOpenChannel(videoInfo.byBoardIndex, videoInfo.byChannelIndex);
  if (channel == nullptr) {
    Logger::get(Logger::Error) << "Magewell : Open channel error" << std::endl;
    return nullptr;
  }

  Logger::get(Logger::Info) << "Magewell : Open channel - BoardIndex = " << (int)videoInfo.byBoardIndex
                            << ", ChannelIndex = " << (int)videoInfo.byChannelIndex << std::endl;
  Logger::get(Logger::Info) << "Magewell : Product Name: " << videoInfo.szProductName << std::endl;
  Logger::get(Logger::Info) << "Magewell : Board SerialNo: " << videoInfo.szBoardSerialNo << std::endl;

  MagewellReader* reader = new MagewellReader(id, width, height, channel, width * height * bytesPerPixel, frameRate,
                                              pixelFormat, bytesPerPixel, name);
  if (reader->init() == true) {
    return reader;
  } else {
    return nullptr;
  }
}

bool MagewellReader::init() {
  format = VideoStitch::Magewell::xiColorFormat(getSpec().format);
  if (MWStartVideoCapture(channel, captureEvent)) {
    Logger::get(Logger::Error) << "Magewell : Open Video Capture error" << std::endl;
    return false;
  }
  MWGetVideoBufferInfo(channel, &videoBufferInfo);
  MWGetVideoFrameInfo(channel, videoBufferInfo.iNewestBufferedFullFrame, &videoFrameInfo);
  MWGetVideoSignalStatus(channel, &videoSignalStatus);
  switch (videoSignalStatus.state) {
    case MWCAP_VIDEO_SIGNAL_NONE:
      Logger::get(Logger::Info) << "Magewell : Input signal status: NONE" << std::endl;
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
      Logger::get(Logger::Info) << "Magewell : Input signal fps: " << fps << std::endl;
      Logger::get(Logger::Info) << "Magewell : Input signal interlaced: "
                                << (videoSignalStatus.bInterlaced ? "true" : "false") << std::endl;
      Logger::get(Logger::Info) << "Magewell : Input signal colorspace: ";
      switch (videoSignalStatus.colorFormat) {
        case MWCAP_VIDEO_COLOR_FORMAT_UNKNOWN:
          Logger::get(Logger::Info) << "unknown";
          break;
        case MWCAP_VIDEO_COLOR_FORMAT_RGB:
          Logger::get(Logger::Info) << "RGB";
          break;
        case MWCAP_VIDEO_COLOR_FORMAT_YUV601:
          Logger::get(Logger::Info) << "YUV601";
          break;
        case MWCAP_VIDEO_COLOR_FORMAT_YUV709:
          Logger::get(Logger::Info) << "YUV709";
          break;
        case MWCAP_VIDEO_COLOR_FORMAT_YUV2020:
          Logger::get(Logger::Info) << "YUV2020";
          break;
        case MWCAP_VIDEO_COLOR_FORMAT_YUV2020C:
          Logger::get(Logger::Info) << "YUV2020C";
          break;
      }
      Logger::get(Logger::Info) << std::endl;
      break;
  }
  HNOTIFY notify = MWRegisterNotify(channel, notifyEvent, MWCAP_NOTIFY_VIDEO_FRAME_BUFFERED);
  if (notify == nullptr) {
    Logger::get(Logger::Error) << "Magewell : Register Notify error" << std::endl;
    return false;
  }
  return true;
}

// --------------------------------- Reader implementation ------------------------

ReadStatus MagewellReader::readFrame(mtime_t& date, unsigned char* video) {
  if (WaitForSingleObject(notifyEvent, 2000) == WAIT_TIMEOUT) {
    date = 0;
    return ReadStatus::OK();
  }
  MWGetVideoBufferInfo(channel, &videoBufferInfo);
  MWGetVideoFrameInfo(channel, videoBufferInfo.iNewestBufferedFullFrame, &videoFrameInfo);
  MWCaptureVideoFrameToVirtualAddress(
      channel, videoBufferInfo.iNewestBufferedFullFrame, (LPBYTE)video,
      (DWORD)(VideoReader::getSpec().width * VideoReader::getSpec().height * bytesPerPixel),
      (DWORD)(VideoReader::getSpec().width * bytesPerPixel), false, nullptr, format, (int)VideoReader::getSpec().width,
      (int)getSpec().height);
  WaitForSingleObject(captureEvent, INFINITE);
  date = (mtime_t)(videoFrameInfo.allFieldStartTimes[0] / 10);  // time unit is 100 ns
  Logger::get(Logger::Verbose) << "Magewell : Captured frame "
                               << (int)((videoFrameInfo.allFieldStartTimes[0] * getSpec().frameRate.num) /
                                        (10000000.0 * getSpec().frameRate.den))
                               << std::endl;  // time unit is 100 ns
  return ReadStatus::OK();
}

Status MagewellReader::seekFrame(frameid_t) { return Status::OK(); }
}  // namespace Input
}  // namespace VideoStitch
