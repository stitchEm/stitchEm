// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "codecfactory.hpp"
#include "mpeg2codec.hpp"
#include "mpeg4codec.hpp"
#include "mjpegcodec.hpp"
#include "jpegcodec.hpp"
#include "tiffcodec.hpp"
#include "h264codec.hpp"
#include "prorescodec.hpp"
#include "trivialcodecs.hpp"

Codec* CodecFactory::create(const QString& key, QWidget* parent) {
  Codec* codec = nullptr;
  if (key == "mpeg4") {
    codec = new Mpeg4Codec(parent);
  } else if (key == "mjpeg") {
    codec = new MjpegCodec(parent);
  } else if (key == "mpeg2") {
    codec = new Mpeg2Codec(parent);
  } else if (key == "h264") {
    codec = new H264Codec(parent);
  } else if (key == "jpg") {
    codec = new JpegCodec(parent);
  } else if (key == "tif") {
    codec = new TiffCodec(parent);
  } else if (key == "pam") {
    codec = new PamCodec(parent);
  } else if (key == "ppm") {
    codec = new PpmCodec(parent);
  } else if (key == "png") {
    codec = new PngCodec(parent);
  } else if (key == "null") {
    codec = new NullCodec(parent);
  } else if (key == "yuv420p") {
    codec = new Yuv420Codec(parent);
  } else if (key == "raw") {
    codec = new RawCodec(parent);
  } else if (key == "prores") {
    codec = new ProResCodec(parent);
  }
  if (codec) {
    codec->setup();
  }
  return codec;
}
