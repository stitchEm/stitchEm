// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ximea_reader.hpp"
#include "export.hpp"

extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Input::Reader>* __cdecl createReaderFn(
    const VideoStitch::Ptv::Value* config, VideoStitch::Plugin::VSReaderPlugin::Config runtime) {
  VideoStitch::Input::XimeaReader* reader =
      VideoStitch::Input::XimeaReader::create(config, runtime.width, runtime.height);
  if (reader) {
    return new VideoStitch::Potential<VideoStitch::Input::Reader>(reader);
  }
  return new VideoStitch::Potential<VideoStitch::Input::Reader>(
      VideoStitch::Origin::Input, VideoStitch::ErrType::InvalidConfiguration, "Could not create Ximea reader");
}

extern "C" VS_PLUGINS_EXPORT bool __cdecl handleReaderFn(const VideoStitch::Ptv::Value* config) {
  return VideoStitch::Input::XimeaReader::handles(config);
}
