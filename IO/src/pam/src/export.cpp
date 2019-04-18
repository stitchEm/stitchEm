// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "export.hpp"
#include "pamOutput.hpp"
#include "libgpudiscovery/delayLoad.hpp"
#include <ostream>

#ifdef DELAY_LOAD_ENABLED
SET_DELAY_LOAD_HOOK
#endif  // DELAY_LOAD_ENABLED

/** \name Services for writer plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Output::Output>* createWriterFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSWriterPlugin::Config run_time) {
  VideoStitch::Output::Output* pamWriter = VideoStitch::Output::PamWriter::create(config, run_time);
  if (pamWriter) {
    return new VideoStitch::Potential<VideoStitch::Output::Output>(pamWriter);
  }
  return new VideoStitch::Potential<VideoStitch::Output::Output>(
      VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, "Could not create Pam writer");
}

extern "C" VS_PLUGINS_EXPORT bool handleWriterFn(VideoStitch::Ptv::Value const* config) {
  return VideoStitch::Output::PamWriter::handles(config);
}
//\}
