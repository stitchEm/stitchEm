// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "export.hpp"
#include "portAudioReader.hpp"
#include "portAudioWriter.hpp"
#include "../include/portAudioDiscovery.hpp"

#include "libgpudiscovery/delayLoad.hpp"

#include "libvideostitch/plugin.hpp"
#include "libvideostitch/ptv.hpp"

#ifdef DELAY_LOAD_ENABLED
SET_DELAY_LOAD_HOOK
#endif  // DELAY_LOAD_ENABLED

/** \name Services for reader plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Input::Reader>* createReaderFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSReaderPlugin::Config runtime) {
  VideoStitch::Input::PortAudioReader* reader = VideoStitch::Input::PortAudioReader::create(runtime.id, config);
  if (reader) {
    return new VideoStitch::Potential<VideoStitch::Input::Reader>(reader);
  }
  return new VideoStitch::Potential<VideoStitch::Input::Reader>(
      VideoStitch::Origin::Input, VideoStitch::ErrType::InvalidConfiguration, "Could not create PortAudio reader");
}

extern "C" VS_PLUGINS_EXPORT bool handleReaderFn(VideoStitch::Ptv::Value const* config) {
  return VideoStitch::Input::PortAudioReader::handles(config);
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Plugin::VSDiscoveryPlugin* discoverFn() {
  return VideoStitch::Plugin::PortAudioDiscovery::create();
}

/*Output*/
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Output::Output>* createWriterFn(
    const VideoStitch::Ptv::Value* config, VideoStitch::Plugin::VSWriterPlugin::Config runtime) {
  auto portaudio = VideoStitch::Output::PortAudioWriter::create(config, runtime);
  if (portaudio.ok()) {
    return new VideoStitch::Potential<VideoStitch::Output::Output>(portaudio.release());
  } else {
    return new VideoStitch::Potential<VideoStitch::Output::Output>(portaudio.status());
  }
}

extern "C" VS_PLUGINS_EXPORT bool handleWriterFn(const VideoStitch::Ptv::Value* config) {
  return VideoStitch::Output::PortAudioWriter::handles(config);
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
  return 0;
}
#endif
