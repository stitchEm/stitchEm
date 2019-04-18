// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "v4l2.hpp"
#include "export.hpp"

extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Input::Reader>* createReaderFn(
    const VideoStitch::Ptv::Value* config, VideoStitch::Plugin::VSReaderPlugin::Config runtime) {
  VideoStitch::Input::V4L2Reader* reader = VideoStitch::Input::V4L2Reader::create(config, runtime);
  if (reader) {
    return new VideoStitch::Potential<VideoStitch::Input::Reader>(reader);
  }
  return new VideoStitch::Potential<VideoStitch::Input::Reader>(
      VideoStitch::Origin::Input, VideoStitch::ErrType::InvalidConfiguration, "Could not create reader");
}

extern "C" VS_PLUGINS_EXPORT bool handleReaderFn(const VideoStitch::Ptv::Value* config) {
  return VideoStitch::Input::V4L2Reader::handles(config);
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Plugin::VSDiscoveryPlugin* discoverFn() {
  return VideoStitch::Plugin::V4L2Discovery::create();
}
