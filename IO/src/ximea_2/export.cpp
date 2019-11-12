// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "export.hpp"

#include "libgpudiscovery/delayLoad.hpp"

#include "libvideostitch/plugin.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/logging.hpp"

#include "ximeaDiscovery.hpp"
#include "ximeaReader.hpp"

#ifdef DELAY_LOAD_ENABLED
SET_DELAY_LOAD_HOOK
#endif  // DELAY_LOAD_ENABLED

extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Input::Reader>* __cdecl createReaderFn(
    const VideoStitch::Ptv::Value* config, VideoStitch::Plugin::VSReaderPlugin::Config runtime) {
  VideoStitch::Input::XimeaReader* ximeaReader =
      VideoStitch::Input::XimeaReader::create(runtime.id, config, runtime.width, runtime.height);
  if (ximeaReader) {
    return new VideoStitch::Potential<VideoStitch::Input::Reader>(ximeaReader);
  }
  return new VideoStitch::Potential<VideoStitch::Input::Reader>(
      VideoStitch::Origin::Input, VideoStitch::ErrType::InvalidConfiguration, "Could not create Ximea reader");
}

extern "C" VS_PLUGINS_EXPORT bool __cdecl handleReaderFn(const VideoStitch::Ptv::Value* config) {
  return config && config->has("type") && config->has("type")->asString() == "ximea";
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Output::Output>* createWriterFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSWriterPlugin::Config run_time) {
  return new VideoStitch::Potential<VideoStitch::Output::Output>(
      VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, "Could not create Ximea writer");
}

extern "C" VS_PLUGINS_EXPORT bool handleWriterFn(VideoStitch::Ptv::Value const* config) {
  bool lReturn(false);

  // TODOLATERSTATUS propagate config problem
  VideoStitch::Logger::get(VideoStitch::Logger::Verbose) << "Invalid ximea config encountered" << std::endl;

  return lReturn;
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Plugin::VSDiscoveryPlugin* discoverFn() {
  return VideoStitch::Plugin::XimeaDiscovery::create();
}
