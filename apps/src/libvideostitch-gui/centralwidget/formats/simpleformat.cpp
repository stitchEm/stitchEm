// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "simpleformat.hpp"

SimpleFormat::SimpleFormat(QString theFormat, ExtensionHandler* theHandler, QWidget* const parent)
    : Format(parent), format(theFormat) {
  supportedCodecs << format;
  handler = theHandler;
  setCodec(format);
}

VideoStitch::Ptv::Value* SimpleFormat::getOutputConfig() const {
  VideoStitch::Ptv::Value* outputConfig = nullptr;
  if (codec) {
    outputConfig = codec->getOutputConfig();
    outputConfig->get("type")->asString() = format.toStdString();
  }
  return outputConfig;
}

bool SimpleFormat::setFromOutputConfig(const VideoStitch::Ptv::Value* config) {
  setCodec(format);
  return codec->setFromOutputConfig(config);
}

bool SimpleFormat::isACodecToo() const { return true; }
