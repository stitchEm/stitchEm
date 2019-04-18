// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "decklink_reader.hpp"
#include "decklink_helpers.hpp"

#include "frameRateHelpers.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

#include <locale>

namespace VideoStitch {
namespace Input {

// ------------------------ Lifecycle --------------------------------

DeckLinkReader::DeckLinkReader(readerid_t id, int64_t width, int64_t height, PixelFormat pixelFormat,
                               Plugin::DisplayMode displayMode, int64_t frameSize, FrameRate frameRate,
                               std::shared_ptr<IDeckLink> subDevice, std::shared_ptr<IDeckLinkInput> input,
                               std::string name)
    : Reader(id),
      VideoReader(width, height, frameSize, pixelFormat, Host, frameRate, 0, NO_LAST_FRAME, false /* not procedural */,
                  nullptr),
      subDevice(subDevice),
      configurationForHalfDuplex(DeckLink::configureDuplexMode(subDevice)),
      input(input),
      name(name),
      displayMode(displayMode),
      frameAvailable(false) {
  currentVideoFrame.resize((size_t)getFrameDataSize());
  memset(currentVideoFrame.data(), 0, currentVideoFrame.size());
}

DeckLinkReader::~DeckLinkReader() { input->SetCallback(nullptr); }

// -------------------------- Plugin implementation ----------------------------------

DeckLinkReader* DeckLinkReader::create(readerid_t id, const Ptv::Value* config, const int64_t width,
                                       const int64_t height) {
  // Get the configuration
  std::string name, pixel_format;
  bool interleaved = false;
  FrameRate frameRate(-1, 1);
  if (Parse::populateString("Decklink reader", *config, "name", name, true) != VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "Error! Missing DeckLink device name (\"name\" field) in the configuration."
                               << std::endl;
    return nullptr;
  }
  if (Parse::populateBool("DeckLink reader", *config, "interleaved", interleaved, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error)
        << "Error! Missing DeckLink interleaved mode (\"interleaved\" field) in the configuration of the " << name
        << " device." << std::endl;
    return nullptr;
  }
  if (!config->has("frame_rate")) {
    double fps = 0.0;
    if (Parse::populateDouble("DeckLink reader", *config, "fps", fps, false) != VideoStitch::Parse::PopulateResult_Ok) {
      Logger::get(Logger::Error)
          << "Error! Missing DeckLink frame rate (\"frame_rate\") or fps (\"fps\") in the configuration of the " << name
          << " device." << std::endl;
      return nullptr;
    }
    frameRate = Util::fpsToNumDen(fps);
  } else {
    const Ptv::Value* fpsConf = config->has("frame_rate");
    if ((Parse::populateInt("DeckLink reader", *fpsConf, "num", frameRate.num, false) !=
         VideoStitch::Parse::PopulateResult_Ok) ||
        (Parse::populateInt("DeckLink reader", *fpsConf, "den", frameRate.den, false) !=
         VideoStitch::Parse::PopulateResult_Ok)) {
      Logger::get(Logger::Error)
          << "Error! DeckLink frame rate (\"frame_rate\") couldn't be retrieved in the configuration of the " << name
          << " device." << std::endl;
      return nullptr;
    }
  }
  if (Parse::populateString("DeckLink reader", *config, "pixel_format", pixel_format, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error)
        << "Error! Missing DeckLink pixel format (\"pixel_format\" field) in the configuration of the " << name
        << " device." << std::endl;
    return nullptr;
  }

  {
    // Get the display mode, the pixel format and the frame size
    const BMDDisplayMode bmdDisplayMode =
        DeckLink::Helpers::getInstance().bmdDisplayMode(width, height, interleaved, frameRate);
    const BMDPixelFormat bmdPixelFormat = DeckLink::Helpers::getInstance().bmdPixelFormat(pixel_format);
    const Plugin::DisplayMode displayMode = DeckLink::Helpers::getInstance().displayMode(bmdDisplayMode);
    const PixelFormat pixelFormat = DeckLink::Helpers::getInstance().pixelFormat(bmdPixelFormat);
    const int64_t frameSize = DeckLink::frameSize(bmdPixelFormat, width, height);

    if (frameSize == 0) {
      Logger::get(Logger::Error) << "Error! DeckLink API could not calculate the frame size for the " << name
                                 << " device. Please check the pixel format in the device's configuration."
                                 << std::endl;
      return nullptr;
    }

    // Get the sub device
    Logger::get(Logger::Debug) << "DeckLink API: accessing the " << name << " device." << std::endl;
    std::shared_ptr<IDeckLink> subDevice = VideoStitch::DeckLink::retrieveDevice(name);
    if (!subDevice) {
      Logger::get(Logger::Error) << "Error! DeckLink API could not find the " << name << " device." << std::endl;
      return nullptr;
    }

    IDeckLinkInput* tempInput = nullptr;
    if (subDevice->QueryInterface(IID_IDeckLinkInput, (void**)&tempInput) != S_OK) {
      Logger::get(Logger::Error) << "Error! The " << name << " device does not accept streaming capture, output only."
                                 << std::endl;
      return nullptr;
    }
    std::shared_ptr<IDeckLinkInput> input(tempInput, [](IDeckLinkInput* p) {
      if (p) {
        p->StopStreams();
        p->SetCallback(nullptr);
        p->DisableVideoInput();
        p->DisableAudioInput();
        p->Release();
      }
    });

    // Check the configuration
    BMDDisplayModeSupport displayModeSupport;
    BMDVideoInputFlags videoInputFlags = bmdVideoInputFlagDefault;
    videoInputFlags |= bmdVideoInputEnableFormatDetection;

    HRESULT result =
        input->DoesSupportVideoMode(bmdDisplayMode, bmdPixelFormat, videoInputFlags, &displayModeSupport, nullptr);
    if (result != S_OK || displayModeSupport == bmdDisplayModeNotSupported) {
      Logger::get(Logger::Error) << "Error! The " << name << " device does not support this input display mode: "
                                 << DeckLink::videoModeToString(displayMode, pixelFormat) << "." << std::endl;
      return nullptr;
    }
    if (displayModeSupport == bmdDisplayModeSupportedWithConversion) {
      Logger::get(Logger::Warning) << "DeckLink API: the given input display mode is supported with conversion on the "
                                   << name << " device." << std::endl;
    }
    Logger::get(Logger::Debug) << "DeckLink API: the given input display mode is supported on the " << name
                               << " device." << std::endl;

    // Install callbacks
    DeckLinkReader* reader = new DeckLinkReader(
        id, width, height, DeckLink::Helpers::getInstance().pixelFormat(bmdPixelFormat),
        DeckLink::Helpers::getInstance().displayMode(bmdDisplayMode), frameSize, frameRate, subDevice, input, name);
    input->SetCallback(reader);

    // Configure the device
    result = input->EnableVideoInput(bmdDisplayMode, bmdPixelFormat, videoInputFlags);
    if (result != S_OK) {
      switch (result) {
        case E_FAIL:
          Logger::get(Logger::Error) << "Error! DeckLink API could not enable the video capture on the " << name
                                     << " device." << std::endl;
          break;
        case E_ACCESSDENIED:
          Logger::get(Logger::Error) << "Error! DeckLink API could not enable the video capture on the " << name
                                     << " device: hardware is not accessible or input stream is currently active."
                                     << std::endl;
          break;
        case E_OUTOFMEMORY:
          Logger::get(Logger::Error) << "Error! DeckLink API could not enable the video capture on the " << name
                                     << " device: out of memory." << std::endl;
          break;
        default:
          Logger::get(Logger::Error) << "Error! DeckLink API could not enable the video capture on the " << name
                                     << " unknown error." << std::endl;
          break;
      }
      return nullptr;
    }

    // Start the capture
    Logger::get(Logger::Debug) << "DeckLink API: start synchronized capture on the " << name << " device." << std::endl;
    result = input->StartStreams();
    if (result != S_OK) {
      switch (result) {
        case E_ACCESSDENIED:
          Logger::get(Logger::Error) << "Error! The " << name << " device's input stream is already running."
                                     << std::endl;
          break;
        case E_UNEXPECTED:
          Logger::get(Logger::Error) << "Error! The " << name << " device has no input enabled." << std::endl;
          break;
      }
      return nullptr;
    }
    return reader;
  }
}

bool DeckLinkReader::handles(const Ptv::Value* config) {
  return config && config->has("type") && (config->has("type")->asString() == "decklink");
}

// --------------------------------- Reader implementation ------------------------

ReadStatus DeckLinkReader::readFrame(mtime_t& timestamp, unsigned char* videoFrame) {
  std::unique_lock<std::mutex> lk(m);
  cv.wait(lk, [this] { return frameAvailable; });

  memcpy(videoFrame, currentVideoFrame.data(), currentVideoFrame.size());
  timestamp = videoTimeStamp;  // value from decklink in µs
  frameAvailable = false;

  return ReadStatus::OK();
}

Status DeckLinkReader::seekFrame(frameid_t) { return Status::OK(); }

// ----------------------------------- Callback implementation ----------------------

HRESULT DeckLinkReader::VideoInputFrameArrived(IDeckLinkVideoInputFrame* videoFrame, IDeckLinkAudioInputPacket*) {
  std::lock_guard<std::mutex> lk(m);

  if (videoFrame == nullptr) {
    Logger::get(Logger::Error) << "Error! Frame is invalid on the " << name
                               << " device. Please check the input display format set on the device."
                               << " Expects " << DeckLink::videoModeToString(displayMode, getSpec().format)
                               << std::endl;
    return E_FAIL;
  }

  if (videoFrame->GetFlags() == (unsigned)bmdFrameHasNoInputSource) {
    Logger::get(Logger::Verbose) << "No input signal from device" << name << std::endl;
  }

  // Video capture
  void* frame;
  if (videoFrame->GetBytes((void**)&frame) != S_OK) {
    return E_FAIL;
  }
  memcpy(currentVideoFrame.data(), frame, currentVideoFrame.size());

  /*cf Decklink doc page 292*/
  BMDTimeValue frameTime, frameDuration;
  videoFrame->GetStreamTime(&frameTime, &frameDuration, (BMDTimeScale)1000000);  // timestamp in µs
  videoTimeStamp = (mtime_t)frameTime;
  frameAvailable = true;
  cv.notify_one();
  return S_OK;
}

HRESULT DeckLinkReader::VideoInputFormatChanged(BMDVideoInputFormatChangedEvents notificationEvents,
                                                IDeckLinkDisplayMode* newDisplayMode,
                                                BMDDetectedVideoInputFormatFlags flags) {
  // Display mode changed
  if (notificationEvents & bmdVideoInputDisplayModeChanged) {
    const Plugin::DisplayMode _newDisplayMode =
        DeckLink::Helpers::getInstance().displayMode(newDisplayMode->GetDisplayMode());
    Logger::get(Logger::Info) << "DeckLink API: Input signal video display mode changed to: "
                              << DeckLink::displayModeToString(_newDisplayMode) << " on device: " << name << std::endl;
  }

  // Field dominance changed: see section 2.7.5 Decklink API doc
  if (notificationEvents & bmdVideoInputFieldDominanceChanged) {
    const std::string fieldDominance = DeckLink::fieldDominanceToString(newDisplayMode->GetFieldDominance());
    Logger::get(Logger::Info) << "DeckLink API: Input signal field dominance changed to: " << fieldDominance
                              << " on device: " << name << std::endl;
  }

  // Color space changed: see section 2.7.24 Decklink API doc
  if (notificationEvents & bmdVideoInputColorspaceChanged) {
    const std::string colorSpace = DeckLink::colotSpaceToString(flags);
    Logger::get(Logger::Info) << "DeckLink API: Input signal pixel format changed to: " << colorSpace
                              << " on device: " << name << std::endl;
  }
  return S_OK;
}

}  // namespace Input
}  // namespace VideoStitch
