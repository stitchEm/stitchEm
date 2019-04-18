// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "decklink_reader.hpp"
#include "decklink_writer.hpp"
#include "decklink_discovery.hpp"
#include "export.hpp"

#include "libgpudiscovery/delayLoad.hpp"

#ifdef DELAY_LOAD_ENABLED
SET_DELAY_LOAD_HOOK
#endif  // DELAY_LOAD_ENABLED

extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Input::Reader>* createReaderFn(
    const VideoStitch::Ptv::Value* config, VideoStitch::Plugin::VSReaderPlugin::Config runtime) {
  VideoStitch::Input::DeckLinkReader* decklinkReader =
      VideoStitch::Input::DeckLinkReader::create(runtime.id, config, runtime.width, runtime.height);
  if (decklinkReader) {
    return new VideoStitch::Potential<VideoStitch::Input::Reader>(decklinkReader);
  }
  return new VideoStitch::Potential<VideoStitch::Input::Reader>(
      VideoStitch::Origin::Input, VideoStitch::ErrType::InvalidConfiguration, "Could not create Decklink reader");
}

extern "C" VS_PLUGINS_EXPORT bool handleReaderFn(const VideoStitch::Ptv::Value* config) {
  return VideoStitch::Input::DeckLinkReader::handles(config);
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Output::Output>* createWriterFn(
    const VideoStitch::Ptv::Value* config, VideoStitch::Plugin::VSWriterPlugin::Config runtime) {
  VideoStitch::Output::Output* decklinkWriter = VideoStitch::Output::DeckLinkWriter::create(
      config, runtime.name, runtime.width, runtime.height, runtime.framerate, runtime.depth, runtime.layout);
  if (decklinkWriter) {
    return new VideoStitch::Potential<VideoStitch::Output::Output>(decklinkWriter);
  }
  return new VideoStitch::Potential<VideoStitch::Output::Output>(
      VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, "Could not create Decklink writer");
}
extern "C" VS_PLUGINS_EXPORT bool handleWriterFn(const VideoStitch::Ptv::Value* config) {
  return VideoStitch::Output::DeckLinkWriter::handles(config);
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Plugin::VSDiscoveryPlugin* discoverFn() {
  return VideoStitch::Plugin::DeckLinkDiscovery::create();
}
