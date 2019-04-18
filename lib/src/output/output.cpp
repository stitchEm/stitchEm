// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "discardOutput.hpp"
#include "profilingOutput.hpp"

#include "util/plugin.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/stitchOutput.hpp"

#include <cassert>
#include <iostream>
#include <mutex>

namespace VideoStitch {

using namespace Plugin;

namespace Output {

Output::~Output() {}

Output::Output(const std::string& nameParam) {
  strncpy(name, nameParam.c_str(), sizeof(name));
  name[sizeof(name) - 1] = '\0';
}

VideoWriter* Output::getVideoWriter() const { return dynamic_cast<VideoWriter*>(const_cast<Output*>(this)); }

AudioWriter* Output::getAudioWriter() const {
  AudioWriter* audio = dynamic_cast<AudioWriter*>(const_cast<Output*>(this));
  if (audio) {
    if (audio->getChannelLayout() != Audio::UNKNOWN && audio->getSamplingRate() != Audio::SamplingRate::SR_NONE &&
        audio->getSamplingDepth() != Audio::SamplingDepth::SD_NONE) {
      return audio;
    }
  }
  return nullptr;
}

BaseConfig::BaseConfig() : numberNumDigits(1), downsamplingFactor(1) {}

/**
 * Returns true on success.
 */
Status BaseConfig::parse(const Ptv::Value& config) {
  clear();
  // Make sure config is an object.
  if (!Parse::checkType("Output", config, Ptv::Value::OBJECT)) {
    return {Origin::Output, ErrType::InvalidConfiguration, "'OutputWriter' value has wrong type"};
  }
  std::string strFmtStr;
  Parse::PopulateResult popResFmt = Parse::populateString("Output", config, "type", strFmtStr, true);
  strncpy(strFmt, strFmtStr.c_str(), sizeof(strFmt));
  strFmt[sizeof(strFmt) - 1] = '\0';
  if (popResFmt == Parse::PopulateResult_WrongType) {
    return {Origin::Output, ErrType::InvalidConfiguration, "OutputWriter 'type' must be a string"};
  }
  std::string baseNameStr;
  Parse::PopulateResult popResBasename = Parse::populateString("Output", config, "filename", baseNameStr, true);
  strncpy(baseName, baseNameStr.c_str(), sizeof(baseName));
  baseName[sizeof(baseName) - 1] = '\0';
  if (popResBasename == Parse::PopulateResult_WrongType) {
    return {Origin::Output, ErrType::InvalidConfiguration, "OutputWriter 'filename' must be a string"};
  }
  if (Parse::populateInt("Output", config, "numbered_digits", numberNumDigits, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Output, ErrType::InvalidConfiguration, "OutputWriter 'numbered_digits' must be an integer value"};
  }
  if (Parse::populateInt("Output", config, "downsampling_factor", downsamplingFactor, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Output, ErrType::InvalidConfiguration,
            "OutputWriter 'downsampling_factor' must be an integer value"};
  }
  if (baseNameStr == "") {
    return {Origin::Output, ErrType::InvalidConfiguration, "OutputWriter output filename is present but empty"};
  }
  return Status::OK();
}

void BaseConfig::clear() {
  strFmt[0] = '\0';
  baseName[0] = '\0';
  numberNumDigits = 1;
  downsamplingFactor = 1;
}

OutputEventManager& Output::getOutputEventManager() { return outputEventManager; }

VideoWriter::VideoWriter(unsigned width, unsigned height, FrameRate framerate, VideoStitch::PixelFormat pixelFormat,
                         AddressSpace outputType)
    : Output(""),
      latency(0),
      width(width),
      height(height),
      framerate(framerate),
      pixelFormat(pixelFormat),
      outputType(outputType) {}

AudioWriter::AudioWriter(Audio::SamplingRate rate, Audio::SamplingDepth depth, Audio::ChannelLayout layout)
    : Output(""), rate(rate), depth(depth), layout(layout) {}

VideoWriter::~VideoWriter() {}

AudioWriter::~AudioWriter() {}

Potential<Output> create(const Ptv::Value& config, const std::string& name, unsigned width, unsigned height,
                         FrameRate framerate, Audio::SamplingRate rate, Audio::SamplingDepth depth,
                         Audio::ChannelLayout audioLayout) {
  BaseConfig baseConfig;
  FAIL_CAUSE(baseConfig.parse(config), Origin::Output, ErrType::InvalidConfiguration,
             "Cannot create output writer '" + name + "'");

  // If we're given a downsampling factor, make sure the factor is doable and modify width/height/padding.
  if (baseConfig.downsamplingFactor > 1) {
    if ((width % baseConfig.downsamplingFactor != 0) || (height % baseConfig.downsamplingFactor != 0)) {
      return {Origin::Output, ErrType::InvalidConfiguration,
              "Ouput writer '" + name +
                  "' specified an invalid downsampling_factor. The output dimensions must be a multiple of the "
                  "downsampling factor."};
    }
    width /= baseConfig.downsamplingFactor;
    height /= baseConfig.downsamplingFactor;
  }

  // First try to open with plugins.
  {
    std::unique_lock<std::mutex> lock(pluginsMutex);
    for (VSWriterPlugin::InstanceVector::const_iterator l_it = VSWriterPlugin::Instances().begin(),
                                                        l_last = VSWriterPlugin::Instances().end();
         l_it != l_last; ++l_it) {
      if ((*l_it)->handles(&config)) {
        Potential<Output>* potWriter =
            (*l_it)->create(&config, VSWriterPlugin::Config(name, width, height, framerate, rate, depth, audioLayout));
        if (!potWriter->ok()) {
          std::stringstream msg;
          const Status writerStatus = potWriter->status();
          delete potWriter;
          msg << "Couldn't create the writer for plugin '" << (*l_it)->getName() << "'";
          return {Origin::Output, ErrType::SetupFailure, msg.str(), writerStatus};
        } else {
          Output* writer = potWriter->release();
          delete potWriter;
          return writer;
        }
      }
    }
  }

  // Without plugins
  if (!strcmp(baseConfig.strFmt, "null")) {
    return Potential<Output>(new DiscardVideoWriter(name, width, height, framerate));
  } else if (!strcmp(baseConfig.strFmt, "profiling")) {
    return Potential<Output>(new ProfilingWriter(name, width, height, framerate));
  } else {
    std::stringstream msg;
    msg << "Could not create an output for configuration '" << std::string(baseConfig.strFmt) << "'";
    if (VSWriterPlugin::Instances().empty()) {
      msg << ". \n"
          << " No output plugin has been loaded. Check your software installation.";
    }
    return {Origin::Output, ErrType::InvalidConfiguration, msg.str()};
  }
}

int64_t VideoWriter::getExpectedFrameSize() const { return getExpectedFrameSizeFor(pixelFormat, width, height); }

int64_t VideoWriter::getExpectedFrameSizeFor(VideoStitch::PixelFormat format, int64_t width, int64_t height) {
  switch (format) {
    case VideoStitch::PixelFormat::RGBA:
    case VideoStitch::PixelFormat::BGRU:
      return width * height * 4;
    case VideoStitch::PixelFormat::RGB:
    case VideoStitch::PixelFormat::BGR:
      return width * height * 3;
    case VideoStitch::PixelFormat::YV12:
    case VideoStitch::PixelFormat::NV12:
      return (width * height * 3) / 2;
    case VideoStitch::PixelFormat::YUV422P10:
      // 20-bits pixels, but padded to 16-bits per component
      return width * height * 4;
    case VideoStitch::PixelFormat::UYVY:
    case VideoStitch::PixelFormat::YUY2:
      return width * height * 2;
    case VideoStitch::PixelFormat::Grayscale:
      return width * height;
    default:
      assert(false);
      return 0;
  }
}

StereoWriter::~StereoWriter() {}
}  // namespace Output
}  // namespace VideoStitch
