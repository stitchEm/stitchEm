// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <export.hpp>
#include "mp4Input.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/plugin.hpp"
#include "libvideostitch/ptv.hpp"
#include <ostream>

/** \name Services for reader plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Input::Reader>* createReaderFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSReaderPlugin::Config runtime) {
  VideoStitch::Input::Mp4Reader* mp4 = VideoStitch::Input::Mp4Reader::create(runtime.id, config->asString(), runtime);
  if (mp4) {
    return new VideoStitch::Potential<VideoStitch::Input::Reader>(mp4);
  }
  return new VideoStitch::Potential<VideoStitch::Input::Reader>(
      VideoStitch::Origin::Input, VideoStitch::ErrType::InvalidConfiguration, "Could not create MP4 reader");
}

extern "C" VS_PLUGINS_EXPORT bool handleReaderFn(VideoStitch::Ptv::Value const* config) {
  return VideoStitch::Input::Mp4Reader::handles(config->asString());
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Input::ProbeResult probeReaderFn(std::string const& p_filename) {
  return VideoStitch::Input::Mp4Reader::probe(p_filename);
}
//\}

#ifdef TestLinking
int main() {
  /** This code is not expected to run: it's just a way to check all
      required symbols will be in library. */
  VideoStitch::Ptv::Value const* config = 0;
  {
    VideoStitch::Plugin::VSReaderPlugin::Config runtime;
    createReaderFn(config, runtime);
  }
  handleReaderFn(config);
  probeReaderFn(std::string());
  VideoStitch::Plugin::VSWriterPlugin::Config runtime;
  createWriterFn(config, runtime);
  handleWriterFn(config);
  return 0;
}
#endif
