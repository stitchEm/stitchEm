// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "export.hpp"

#include "sharedmemorywriter.hpp"

#include "libvideostitch/plugin.hpp"

/** \name Services for writer plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Output::Output>* createWriterFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSWriterPlugin::Config run_time) {
  VideoStitch::Output::Output* shared = VideoStitch::Output::SharedMemoryWriter::create(
      config, run_time.name, run_time.width, run_time.height, run_time.paddingTop, run_time.paddingBottom);
  if (shared) {
    return new VideoStitch::Potential<VideoStitch::Output::Output>(shared);
  }
  return new VideoStitch::Potential<VideoStitch::Output::Output>(
      VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, "Could not create shared writer");
}

extern "C" VS_PLUGINS_EXPORT bool handleWriterFn(VideoStitch::Ptv::Value const* config) {
  return VideoStitch::Output::SharedMemoryWriter::handles(config);
}
//\}
