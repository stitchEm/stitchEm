// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <unordered_set>
#include <string>

#include "rtmpPublisher.hpp"
#include "rtmpClient.hpp"
#include "rtmpDiscovery.hpp"
#include "export.hpp"
#include "libvideostitch/ptv.hpp"

/*Input*/
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Input::Reader>* createReaderFn(
    const VideoStitch::Ptv::Value* config, VideoStitch::Plugin::VSReaderPlugin::Config runtime) {
  VideoStitch::Input::RTMPClient* rtmp =
      VideoStitch::Input::RTMPClient::create(runtime.id, config, runtime.width, runtime.height);
  if (rtmp) {
    return new VideoStitch::Potential<VideoStitch::Input::Reader>(rtmp);
  }
  return new VideoStitch::Potential<VideoStitch::Input::Reader>(
      VideoStitch::Origin::Input, VideoStitch::ErrType::InvalidConfiguration, "Could not create RTMP reader");
}

extern "C" VS_PLUGINS_EXPORT bool handleReaderFn(const VideoStitch::Ptv::Value* config) {
  return config && config->has("type") && (config->has("type")->asString() == "rtmp");
}

/*Output*/
const std::unordered_set<std::string> output_formats = {"rtmp", "youtube"};

extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Output::Output>* createWriterFn(
    const VideoStitch::Ptv::Value* config, VideoStitch::Plugin::VSWriterPlugin::Config runtime) {
  auto rtmp = VideoStitch::Output::RTMPPublisher::create(config, runtime);
  if (rtmp.ok()) {
    return new VideoStitch::Potential<VideoStitch::Output::Output>(rtmp.release());
  } else {
    return new VideoStitch::Potential<VideoStitch::Output::Output>(rtmp.status());
  }
}

extern "C" VS_PLUGINS_EXPORT bool handleWriterFn(const VideoStitch::Ptv::Value* config) {
  VideoStitch::Output::BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    return output_formats.find(baseConfig.strFmt) != output_formats.end();
  }
  return false;
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Plugin::VSDiscoveryPlugin* discoverFn() {
  return VideoStitch::Plugin::RTMPDiscovery::create();
}
