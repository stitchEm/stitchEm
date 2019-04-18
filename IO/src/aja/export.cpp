// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "export.hpp"

#include "libgpudiscovery/delayLoad.hpp"

#include "libvideostitch/plugin.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/logging.hpp"

#include "ntv2plugin.hpp"
#include "ntv2Discovery.hpp"
#include "ntv2Reader.hpp"
#include "ntv2Writer.hpp"

#ifdef DELAY_LOAD_ENABLED
SET_DELAY_LOAD_HOOK
#endif  // DELAY_LOAD_ENABLED

extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Input::Reader>* __cdecl createReaderFn(
    const VideoStitch::Ptv::Value* config, VideoStitch::Plugin::VSReaderPlugin::Config runtime) {
  VideoStitch::Input::NTV2Reader* ntv2Reader =
      VideoStitch::Input::NTV2Reader::create(runtime.id, config, runtime.width, runtime.height);
  if (ntv2Reader) {
    return new VideoStitch::Potential<VideoStitch::Input::Reader>(ntv2Reader);
  }
  return new VideoStitch::Potential<VideoStitch::Input::Reader>(
      VideoStitch::Origin::Input, VideoStitch::ErrType::InvalidConfiguration, "Could not create Aja reader");
}

extern "C" VS_PLUGINS_EXPORT bool __cdecl handleReaderFn(const VideoStitch::Ptv::Value* config) {
  return config && config->has("type") && config->has("type")->asString() == "aja";
}

/** \name Services for writer plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Output::Output>* createWriterFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSWriterPlugin::Config run_time) {
  VideoStitch::Output::Output* lReturn = nullptr;
  VideoStitch::Output::BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    lReturn = VideoStitch::Output::NTV2Writer::create(*config, run_time.name, baseConfig.baseName, run_time.width,
                                                      run_time.height, run_time.framerate);
  }
  if (lReturn) {
    return new VideoStitch::Potential<VideoStitch::Output::Output>(lReturn);
  }
  return new VideoStitch::Potential<VideoStitch::Output::Output>(
      VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, "Could not create Aja writer");
}

extern "C" VS_PLUGINS_EXPORT bool handleWriterFn(VideoStitch::Ptv::Value const* config) {
  bool lReturn(false);
  VideoStitch::Output::BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    lReturn = (!strcmp(baseConfig.strFmt, "aja"));
  } else {
    // TODOLATERSTATUS propagate config problem
    VideoStitch::Logger::get(VideoStitch::Logger::Verbose) << "Invalid aja config encountered" << std::endl;
  }
  return lReturn;
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Plugin::VSDiscoveryPlugin* discoverFn() {
  return VideoStitch::Plugin::Ntv2Discovery::create();
}
