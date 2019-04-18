// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mp4format.hpp"
#include "extensionhandlers/mp4extensionhandler.hpp"
#include "codecs/codec.hpp"
#include "libvideostitch-gui/utils/videocodecs.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"
#include "libvideostitch/parse.hpp"

Mp4Format::Mp4Format(QWidget* const parent) : Format(parent) {
  for (auto codecName :
       VideoStitch::VideoCodec::getSupportedCodecsFor(VideoStitch::OutputFormat::OutputFormatEnum::MP4)) {
    supportedCodecs.append(VideoStitch::VideoCodec::getStringFromEnum(codecName));
  }
  handler = new Mp4ExtensionHandler();
  setCodec(VideoStitch::VideoCodec::getStringFromEnum(VideoStitch::VideoCodec::VideoCodecEnum::H264));
}

VideoStitch::Ptv::Value* Mp4Format::getOutputConfig() const {
  VideoStitch::Ptv::Value* outputConfig = nullptr;
  if (codec) {
    outputConfig = codec->getOutputConfig();
    outputConfig->get("video_codec")->asString() = codec->getKey().toStdString();
    outputConfig->get("type")->asString() = "mp4";
  }
  return outputConfig;
}

bool Mp4Format::setFromOutputConfig(const VideoStitch::Ptv::Value* config) {
  std::string codecString;

  if (VideoStitch::Parse::populateString("Ptv", *config, "video_codec", codecString, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    return false;
  }

  setCodec(QString::fromStdString(codecString));
  return codec != nullptr && codec->setFromOutputConfig(config);
}

bool Mp4Format::isACodecToo() const { return false; }
