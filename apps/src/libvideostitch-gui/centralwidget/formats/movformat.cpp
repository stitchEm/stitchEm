// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "movformat.hpp"
#include "extensionhandlers/movextensionhandler.hpp"
#include "codecs/codec.hpp"
#include "libvideostitch-gui/utils/videocodecs.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"
#include "libvideostitch/parse.hpp"

MovFormat::MovFormat(QWidget* const parent) : Format(parent) {
  for (auto codecNames :
       VideoStitch::VideoCodec::getSupportedCodecsFor(VideoStitch::OutputFormat::OutputFormatEnum::MOV)) {
    supportedCodecs.append(VideoStitch::VideoCodec::getStringFromEnum(codecNames));
  }
  handler = new MovExtensionHandler();
  setCodec(VideoStitch::VideoCodec::getStringFromEnum(VideoStitch::VideoCodec::VideoCodecEnum::H264));
}

VideoStitch::Ptv::Value* MovFormat::getOutputConfig() const {
  VideoStitch::Ptv::Value* outputConfig = nullptr;
  if (codec) {
    outputConfig = codec->getOutputConfig();
    outputConfig->get("video_codec")->asString() = codec->getKey().toStdString();
    outputConfig->get("type")->asString() = "mov";
  }
  return outputConfig;
}

bool MovFormat::setFromOutputConfig(const VideoStitch::Ptv::Value* config) {
  std::string codecString;

  if (VideoStitch::Parse::populateString("Ptv", *config, "video_codec", codecString, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    return false;
  }

  setCodec(QString::fromStdString(codecString));
  return codec != nullptr && codec->setFromOutputConfig(config);
}

bool MovFormat::isACodecToo() const { return false; }
