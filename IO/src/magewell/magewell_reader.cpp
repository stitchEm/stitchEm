// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "magewell_reader.hpp"
#include "magewell_helpers.hpp"
#include "magewell_discovery.hpp"

#include "frameRateHelpers.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

#include <memory>
#include <sstream>
#include <codecvt>

namespace VideoStitch {
namespace Input {

static const BOOL LOW_DELAY_MODE(FALSE);  // Disable internal Magewell buffers

static void shutdown(HANDLE videoCapture) {
  // Stop video capture
  if (videoCapture != NULL) {
    if (XIS_IsVideoCaptureStarted(videoCapture) == TRUE) {
      if (XIS_StopVideoCapture(videoCapture) != TRUE) {
        Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not stop video capture" << std::endl;
      }
    }
    // Close the video capture
    XIS_CloseVideoCapture(videoCapture);
  }
  XIS_Uninitialize();
}

class VideoPropertyGuard {
 public:
  explicit VideoPropertyGuard(HANDLE videoCapture) { videoProperty = XIS_OpenVideoCapturePropertyHandle(videoCapture); }
  ~VideoPropertyGuard() {
    if (videoProperty != NULL) {
      XIP_ClosePropertyHandle(videoProperty);
    }
  }
  HANDLE videoProperty;

 private:
  VideoPropertyGuard(const VideoPropertyGuard&);
  VideoPropertyGuard& operator=(const VideoPropertyGuard&);
};

// ------------------------ Lifecycle --------------------------------

MagewellReader::MagewellReader(readerid_t id, const int64_t width, const int64_t height, HANDLE videoCapture,
                               int64_t frameSize, FrameRate fps, const std::string& name)
    : Reader(id),
      VideoReader(width, height, frameSize, YUY2, Host, fps, 0, NO_LAST_FRAME, false /* not a procedural reader */,
                  nullptr),
      videoCapture(videoCapture),
      videoFrame(NULL),
      name(name),
      frameAvailable(false) {
  videoFrame.resize((size_t)getFrameDataSize());
  memset(videoFrame.data(), 0, videoFrame.size());
}

MagewellReader::~MagewellReader() { shutdown(videoCapture); }

// -------------------------- Plugin implementation ----------------------------------

MagewellReader* MagewellReader::create(readerid_t id, const Ptv::Value* config, const int64_t width,
                                       const int64_t height) {
  // Get the configuration
  std::string name;
  if (Parse::populateString("Magewell reader", *config, "name", name, true) != VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "Error! Magewell library (XIS) camera name (\"name\") couldn't be retrieved. Please, "
                                  "give a name to your input. Aborting."
                               << std::endl;
    return nullptr;
  }

  bool ok = false;
  XIPHD_SCALE_TYPE scaleType = getScaleType(*config, ok);
  if (!ok) {
    return nullptr;
  }

  // VIDEO
  FrameRate frameRate;
  if (!config->has("frame_rate")) {
    double fps = 0.0;
    if (Parse::populateDouble("Magewell reader", *config, "fps", fps, false) != VideoStitch::Parse::PopulateResult_Ok) {
      Logger::get(Logger::Error)
          << "Magewell : frame rate (\"frame_rate\") or fps (\"fps\") couldn't be retrieved. Aborting." << std::endl;
      return nullptr;
    }
    frameRate = Util::fpsToNumDen(fps);
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

  bool interleaved = false;
  if (Parse::populateBool("Magewell reader", *config, "interleaved", interleaved, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Warning)
        << "Warning! Magewell library (XIS) interleaved mode (\"interleaved\") coudldn't be retrieved." << std::endl;
  }

  // Initialize
  if (XIS_Initialize() != TRUE) {
    Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not initialize. Aborting." << std::endl;
    shutdown(NULL);
    return nullptr;
  }

  // Select and open the video device
  ok = false;
  VIDEO_CAPTURE_INFO_EX videoCaptureInfo = retrieveVideoCaptureInfo(name, ok);
  if (!ok) {
    shutdown(NULL);
    return nullptr;
  }

  HANDLE videoCapture = XIS_OpenVideoCapture(videoCaptureInfo.szDShowID);
  if (videoCapture == NULL) {
    Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not initialize. Aborting." << std::endl;
    shutdown(NULL);
    return nullptr;
  }

  // Configure the video capture
  VideoPropertyGuard videoPropertyGuard(videoCapture);
  VideoStitch::Magewell::SupportedMagewellCaptureFamily family =
      VideoStitch::Magewell::retrieveCaptureFamily(videoCaptureInfo);
  const bool isUsb = (family == VideoStitch::Magewell::SupportedMagewellCaptureFamily::UsbCaptureFamily);
  if (!isUsb) {
    if (videoPropertyGuard.videoProperty == NULL) {
      Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not retrieve the property handle. Aborting."
                                 << std::endl;
      shutdown(videoCapture);
      return nullptr;
    }
  }
  if (XIS_SetVideoCaptureBuffering(videoCapture, LOW_DELAY_MODE) != TRUE) {
    Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not set buffering. Aborting." << std::endl;
    shutdown(videoCapture);
    return nullptr;
  }

  if (!isUsb) {
    // Set the zoom type
    if (FAILED(XIPHD_SetScaleType(videoPropertyGuard.videoProperty,
                                  scaleType))) {  // force the capture resolution and remove black borders
      Logger::get(Logger::Error) << "Error! Magewell library (XIPHD) could not set zoom type. Aborting." << std::endl;
      shutdown(videoCapture);
      return nullptr;
    }

    // Set deinterlace type
    if (FAILED(XIPHD_SetDeinterlaceType(videoPropertyGuard.videoProperty,
                                        (interleaved ? XIPHD_DEINTERLACE_BLEND : XIPHD_DEINTERLACE_NONE)))) {
      Logger::get(Logger::Error) << "Error! Magewell library (XIPHD) could not set set the deinterlace type. Aborting."
                                 << std::endl;
      shutdown(videoCapture);
      return nullptr;
    }
  }

  // Set the video capture format
  double fps = double(frameRate.num) / double(frameRate.den);
  if (XIS_SetVideoCaptureFormat(videoCapture, XI_COLOR_YUYV, (int)width, (int)height,
                                (int)std::round(10000000 / fps)) != TRUE) {
    Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not set the video capture format. Aborting."
                               << std::endl;
    shutdown(videoCapture);
    return nullptr;
  }

  if (!isUsb) {
    // XIPHD_SetFlipMode is not impemented with the 2nd generation cards
    HRESULT result = XIPHD_SetFlipMode(videoPropertyGuard.videoProperty, false, false);
    if (FAILED(result) && result != E_NOTIMPL) {
      Logger::get(Logger::Error) << "Error! Magewell library (XIPHD) could not flip the image. Aborting." << std::endl;
      shutdown(videoCapture);
      return nullptr;
    }
    if (FAILED(XIPHD_IsSignalPresent(videoPropertyGuard.videoProperty))) {
      Logger::get(Logger::Error) << "Error! Magewell library (XIPHD) signal not found on device"
                                 << videoCaptureInfo.szDShowID << ". Aborting." << std::endl;
      shutdown(videoCapture);
      return nullptr;
    }
  }
  // Create the reader
  const int64_t frameSize = width * height * 2;
  std::unique_ptr<MagewellReader> reader(
      new MagewellReader(id, width, height, videoCapture, frameSize, frameRate, name));
  if (!reader) {
    shutdown(videoCapture);
    return nullptr;
  }

  // Install callbacks
  if (XIS_SetVideoCaptureCallbackEx(videoCapture, &MagewellReader::videoCallback, reader.get(), LOW_DELAY_MODE) !=
      TRUE) {
    Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not set the video capture callback."
                               << std::endl;
    return nullptr;
  }

  // Start capture
  if (XIS_StartVideoCapture(videoCapture) != TRUE) {
    Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not start the video capture." << std::endl;
    return nullptr;
  }
  return reader.release();
}

bool MagewellReader::handles(const Ptv::Value* config) {
  return config && config->has("type") &&
         (config->has("type")->asString() == "magewell" ||
          config->has("type")->asString() == "magewell_hdmi");  // backward compat
}

// --------------------------------- Reader implementation ------------------------

ReadStatus MagewellReader::readFrame(mtime_t& date, unsigned char* video) {
  std::unique_lock<std::mutex> lock(videoMu);
  videoCv.wait(lock, [this] { return frameAvailable; });

  // Video reading
  memcpy(video, videoFrame.data(), videoFrame.size());
  frameAvailable = false;
  // the time unit in Magewell SDK is 100ns
  date = (mtime_t)(timestamp / 10);

  return ReadStatus::OK();
}

Status MagewellReader::seekFrame(frameid_t /*date*/) { return Status::OK(); }

// --------------------------------- Callback implementation ----------------------

void MagewellReader::videoCallback(const BYTE* pbyImage, int /*cbImageStride*/, void* pvParam, UINT64 u64TimeStamp) {
  MagewellReader* that = static_cast<MagewellReader*>(pvParam);
  {
    std::lock_guard<std::mutex> lock(that->videoMu);
    memcpy(that->videoFrame.data(), pbyImage, that->videoFrame.size());
    that->frameAvailable = true;
    that->timestamp = u64TimeStamp;
  }
  that->videoCv.notify_one();
}

VIDEO_CAPTURE_INFO_EX MagewellReader::retrieveVideoCaptureInfo(const std::string& name, bool& ok) {
  ok = false;
  int nameId = std::stoi(name);
  int nbVideoCaptures = XIS_GetVideoCaptureCount();
  if (nameId >= nbVideoCaptures) {
    Logger::get(Logger::Error) << "Error! Magewell library (XIS): no suitable video device found. Aborting."
                               << std::endl;
    return VIDEO_CAPTURE_INFO_EX();
  }

  VIDEO_CAPTURE_INFO_EX videoCaptureInfo;
  if (XIS_GetVideoCaptureInfoEx(nameId, &videoCaptureInfo) != TRUE) {
    Logger::get(Logger::Error) << "Error! Magewell library (XIS) could not list available video devices. Aborting."
                               << std::endl;
    return VIDEO_CAPTURE_INFO_EX();
  }

  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  std::string deviceName = converter.to_bytes(std::wstring(videoCaptureInfo.szName));
  Logger::get(Logger::Debug) << "Magewell library (XIS): found suitable video capture device id: " << nameId
                             << ", name: " << deviceName << "." << std::endl;
  ok = true;
  return videoCaptureInfo;
}

XIPHD_SCALE_TYPE MagewellReader::getScaleType(const Ptv::Value& config, bool& ok) {
  XIPHD_SCALE_TYPE scaleType = XIPHD_SCALE_DO_NOT_KEEP_ASPECT;
  std::string zoomType;
  if (Parse::populateString("Magewell reader", config, "builtin_zoom", zoomType, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error)
        << "Error! Magewell library (XIS) zoom mode (\"builtin_zoom\") couldn't be retrieved. Aborting." << std::endl;
    ok = false;
    return scaleType;
  }

  if (zoomType == "zoom") {
    scaleType = XIPHD_SCALE_ZOOM_TO_KEEP_ASPECT;
  } else if (zoomType == "fill") {
    scaleType = XIPHD_SCALE_FILL_TO_KEEP_ASPECT;
  } else if (zoomType == "none") {
    scaleType = XIPHD_SCALE_DO_NOT_KEEP_ASPECT;
  } else {
    Logger::get(Logger::Error) << "Error! Magewell library (XIPHD) could not set zoom type: invalid value. Aborting."
                               << std::endl;
    ok = false;
    return scaleType;
  }
  ok = true;
  return scaleType;
}
}  // namespace Input
}  // namespace VideoStitch
