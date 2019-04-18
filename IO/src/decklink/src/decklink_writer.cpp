// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "decklink_writer.hpp"
#include "decklink_helpers.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Output {

const BMDVideoOutputFlags DeckLinkWriter::outputFlags = bmdVideoOutputFlagDefault;
const BMDFrameFlags DeckLinkWriter::outputFrameFlags = bmdFrameFlagDefault;

DeckLinkWriter::DeckLinkWriter(const std::string& name, unsigned width, unsigned height, FrameRate fps,
                               size_t frameSize, const Audio::SamplingDepth depth, const Audio::ChannelLayout layout,
                               std::shared_ptr<IDeckLink> subDevice,
                               std::shared_ptr<IDeckLinkConfiguration> configuration,
                               std::shared_ptr<IDeckLinkConfiguration> configurationForHalfDuplex,
                               std::shared_ptr<IDeckLinkOutput> output,
                               std::shared_ptr<IDeckLinkMutableVideoFrame> outputFrame)
    : Output(name),
      VideoWriter(width, height, fps, UYVY, Host),
      AudioWriter(Audio::SamplingRate::SR_48000, depth, layout),
      subDevice(subDevice),
      configuration(configuration),
      configurationForHalfDuplex(configurationForHalfDuplex),
      output(output),
      outputFrame(outputFrame),
      frameSize(frameSize) {}

DeckLinkWriter::~DeckLinkWriter() {}

DeckLinkWriter* DeckLinkWriter::create(const Ptv::Value* config, const std::string& name, unsigned width,
                                       unsigned height, FrameRate framerate, const Audio::SamplingDepth depth,
                                       const Audio::ChannelLayout layout) {
  // Get the configuration
  int64_t decklink_width = 0;
  int64_t decklink_height = 0;
  bool decklink_interleaved = false;
  FrameRate decklink_frameRate = framerate;
  if (Parse::populateInt("DeckLink writer", *config, "width", decklink_width, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "Error! Missing DeckLink width (\"width\") in configuration." << std::endl;
    return nullptr;
  }
  if (Parse::populateInt("DeckLink writer", *config, "height", decklink_height, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "Error! Missing DeckLink height (\"height\") in configuration." << std::endl;
    return nullptr;
  }
  if (Parse::populateBool("DeckLink writer", *config, "interleaved", decklink_interleaved, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "Error! Missing DeckLink interleaved mode (\"interleaved\") in configuration."
                               << std::endl;
    return nullptr;
  }
  const Ptv::Value* fpsConf = config->has("frame_rate");
  if (fpsConf) {
    if ((Parse::populateInt("DeckLink writer", *fpsConf, "num", decklink_frameRate.num, true) !=
         VideoStitch::Parse::PopulateResult_Ok) ||
        (Parse::populateInt("DeckLink writer", *fpsConf, "den", decklink_frameRate.den, true) !=
         VideoStitch::Parse::PopulateResult_Ok)) {
      Logger::get(Logger::Error) << "Error! Frame rate (\"frame_rate\") couldn't be retrieved in configuration."
                                 << std::endl;
      return nullptr;
    }
  }

  {
    const BMDDisplayMode askedBmdDisplayMode = DeckLink::Helpers::getInstance().bmdDisplayMode(
        decklink_width, decklink_height, decklink_interleaved, decklink_frameRate);
    const Plugin::DisplayMode askedDisplayMode = DeckLink::Helpers::getInstance().displayMode(askedBmdDisplayMode);
    const int64_t frameSize = DeckLink::frameSize(bmdFormat8BitYUV, width, height);

    if (frameSize == 0) {
      Logger::get(Logger::Error) << "Error! DeckLink API could not to calculate the frame size on the device " << name
                                 << "." << std::endl;
      return nullptr;
    }

    // Get the device
    Logger::get(Logger::Debug) << "DeckLink API: accessing the " << name << " device." << std::endl;
    std::shared_ptr<IDeckLink> subDevice = VideoStitch::DeckLink::retrieveDevice(name);
    if (!subDevice) {
      Logger::get(Logger::Error) << "DeckLink API: could not find the device " << name << "." << std::endl;
      return nullptr;
    }

    IDeckLinkOutput* tempOutput = nullptr;
    if (subDevice->QueryInterface(IID_IDeckLinkOutput, (void**)&tempOutput) != S_OK) {
      Logger::get(Logger::Error) << "Error! The device " << name << " can not output stream." << std::endl;
      return nullptr;
    }
    std::shared_ptr<IDeckLinkOutput> output = std::shared_ptr<IDeckLinkOutput>(tempOutput, [](IDeckLinkOutput* p) {
      if (p) {
        p->DisableVideoOutput();
        p->Release();
      }
    });

    IDeckLinkConfiguration* tempConfiguration = nullptr;
    if (subDevice->QueryInterface(IID_IDeckLinkConfiguration, (void**)&tempConfiguration) != S_OK) {
      Logger::get(Logger::Error)
          << "Error! DeckLink API could not to obtain the DeckLinkConfiguration interface on the " << name << " device."
          << std::endl;
      return nullptr;
    }
    std::shared_ptr<IDeckLinkConfiguration> configuration(tempConfiguration,
                                                          VideoStitch::DeckLink::getDefaultDeleter());
    configuration->SetFlag(bmdDeckLinkConfigUse1080pNotPsF, true);

    // Retrieve the configuration for duplex mode, could be different from the previous configuration object
    // We need it before to enable video output
    std::shared_ptr<IDeckLinkConfiguration> configurationForHalfDuplex = DeckLink::configureDuplexMode(subDevice);

    // Configure the device
    BMDDisplayModeSupport displayModeSupport;
    const HRESULT result =
        output->DoesSupportVideoMode(askedBmdDisplayMode, bmdFormat8BitYUV, outputFlags, &displayModeSupport, nullptr);
    if (result == S_OK && displayModeSupport == bmdDisplayModeNotSupported) {
      Logger::get(Logger::Error) << "Error! The device " << name << " does not support this input video mode: "
                                 << DeckLink::videoModeToString(askedDisplayMode, UYVY) << "." << std::endl;
      return nullptr;
    }
    if (displayModeSupport == bmdDisplayModeSupportedWithConversion) {
      Logger::get(Logger::Warning) << "DeckLink API: the device " << name
                                   << " supports with conversion this input video mode: "
                                   << DeckLink::videoModeToString(askedDisplayMode, UYVY) << "." << std::endl;
    }

    switch (output->EnableVideoOutput(askedBmdDisplayMode, outputFlags)) {
      case S_OK:
        break;
      case E_FAIL:
        Logger::get(Logger::Error) << "Error! The device " << name << " failed to enable the video output."
                                   << std::endl;
        return nullptr;
      case E_ACCESSDENIED:
        Logger::get(Logger::Error) << "Error! DeckLink API could not access the hardware on the device " << name
                                   << ". Or output stream is currently active." << std::endl;
        return nullptr;
      case E_OUTOFMEMORY:
        Logger::get(Logger::Error) << "Error! DeckLink API could not enable the video output on the device " << name
                                   << ": out of memory." << std::endl;
        return nullptr;
    }

    if (layout != Audio::UNKNOWN && depth != Audio::SamplingDepth::SD_NONE) {
      BMDAudioSampleType type;
      int chanCount = 0;
      switch (depth) {
        case Audio::SamplingDepth::INT16:
          type = bmdAudioSampleType16bitInteger;
          break;
        case Audio::SamplingDepth::INT32:
          type = bmdAudioSampleType32bitInteger;
          break;
        default:
          Logger::get(Logger::Error) << "Error! DeckLink API invalid sample type." << std::endl;
          return nullptr;
      }
      switch (layout) {
        case Audio::MONO:
          chanCount = 1;
          break;
        case Audio::STEREO:
          chanCount = 2;
          break;
        case Audio::_7POINT1:
          chanCount = 8;
          break;
        default:
          Logger::get(Logger::Error) << "Error! DeckLink API invalid channel layout." << std::endl;
          return nullptr;
      }
      switch (output->EnableAudioOutput(bmdAudioSampleRate48kHz, type, chanCount, bmdAudioOutputStreamTimestamped)) {
        case S_OK:
          break;
        case E_FAIL:
          Logger::get(Logger::Error) << "Error! The device " << name << " failed to enable the audio output."
                                     << std::endl;
          return nullptr;
        case E_ACCESSDENIED:
          Logger::get(Logger::Error) << "Error! DeckLink API could not access the hardware on the device " << name
                                     << ". Or output stream is currently active." << std::endl;
          return nullptr;
        case E_OUTOFMEMORY:
          Logger::get(Logger::Error) << "Error! DeckLink API could not enable the audio output on the device " << name
                                     << ": out of memory." << std::endl;
          return nullptr;
      }
    }

    IDeckLinkMutableVideoFrame* tempOutputFrame = nullptr;
    switch (output->CreateVideoFrame(
        width, height, static_cast<long>(width * DeckLink::Helpers::getInstance().bytesPerPixel(bmdFormat8BitYUV)),
        bmdFormat8BitYUV, outputFrameFlags, &tempOutputFrame)) {
      case S_OK:
        break;
      case E_FAIL:
        Logger::get(Logger::Error) << "Error! DeckLink API failed to create the output frame on the " << name << "."
                                   << std::endl;
        return nullptr;
    }
    std::shared_ptr<IDeckLinkMutableVideoFrame> outputFrame =
        std::shared_ptr<IDeckLinkMutableVideoFrame>(tempOutputFrame, VideoStitch::DeckLink::getDefaultDeleter());

    return new DeckLinkWriter(name, width, height, framerate, frameSize, depth, layout, subDevice, configuration,
                              configurationForHalfDuplex, output, outputFrame);
  }
}

bool DeckLinkWriter::handles(const Ptv::Value* config) {
  return config && config->has("type") && (config->has("type")->asString() == "decklink");
}

void DeckLinkWriter::pushVideo(const Frame& videoFrame) {
  if (outputFrame && output && subDevice) {
    unsigned char* image;
    if (outputFrame->GetBytes((void**)&image) == S_OK) {
      memcpy(image, videoFrame.planes[0], frameSize);
      // TODO: use it when audio is supported.
      // switch (output->ScheduleVideoFrame(outputFrame, date, (1000000 * getFrameRate().den) / getFrameRate().num,
      // 1000000));
      switch (output->DisplayVideoFrameSync(outputFrame.get())) {
        case S_OK:
          break;
        case E_FAIL:
          Logger::get(Logger::Error) << "Error! The device " << getName() << " failed to display the output frame."
                                     << std::endl;
          break;
        case E_ACCESSDENIED:
          Logger::get(Logger::Error) << "Error! The device " << getName()
                                     << " is unable to display the output frame: the video output is not enabled."
                                     << std::endl;
          break;
        case E_INVALIDARG:
          Logger::get(Logger::Error) << "Error! The device " << getName()
                                     << " is unable to display the output frame: the frame attributes are invalid."
                                     << std::endl;
          break;
        case E_OUTOFMEMORY:
          Logger::get(Logger::Error) << "Error! The device " << getName()
                                     << " is unable to display the output frame: too many frames are already scheduled."
                                     << std::endl;
          break;
      }
    }
  }

  if (firstFrame) {
    output->StartScheduledPlayback(videoFrame.pts, 1000000, 1.0);
    firstFrame = false;
  }
}

void DeckLinkWriter::pushAudio(Audio::Samples& audioSamples) {
  if (audioSamples.getNbOfSamples() > 0) {
    unsigned int actualNbSamplesWritten;
    switch (output->ScheduleAudioSamples(audioSamples.getSamples()[0], (unsigned long)audioSamples.getNbOfSamples(),
                                         audioSamples.getTimestamp(), 1000000, &actualNbSamplesWritten)) {
      case S_OK:
        break;
      case E_FAIL:
        Logger::get(Logger::Error) << "Error! The " << getName() << " device failed to output the audio frame."
                                   << std::endl;
        break;
      case E_ACCESSDENIED:
        Logger::get(Logger::Error) << "Error! The " << getName()
                                   << " device failed to output the audio frame. The audio output is not enabled."
                                   << std::endl;
        break;
      case E_INVALIDARG:
        Logger::get(Logger::Error) << "Error! The " << getName()
                                   << " device failed to output the audio frame. No timescale provided." << std::endl;
        break;
    }
  }

  if (firstFrame) {
    output->StartScheduledPlayback(audioSamples.getTimestamp(), 1000000, 1.0);
    firstFrame = false;
  }
}

}  // namespace Output
}  // namespace VideoStitch
