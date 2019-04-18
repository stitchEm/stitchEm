// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "export.hpp"
#include "pngInput.hpp"
#include "pngOutput.hpp"
#include "numberedFilesOutput.hpp"

#include "libgpudiscovery/delayLoad.hpp"

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/plugin.hpp"
#include "libvideostitch/ptv.hpp"
#include <ostream>

#ifdef DELAY_LOAD_ENABLED
SET_DELAY_LOAD_HOOK
#endif  // DELAY_LOAD_ENABLED

/** \name Services for reader plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Input::Reader>* createReaderFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSReaderPlugin::Config runtime) {
  VideoStitch::Input::PngReader* pngReader = VideoStitch::Input::PngReader::create(config, runtime);
  if (pngReader) {
    return new VideoStitch::Potential<VideoStitch::Input::Reader>(pngReader);
  }
  return new VideoStitch::Potential<VideoStitch::Input::Reader>(
      VideoStitch::Origin::Input, VideoStitch::ErrType::InvalidConfiguration, "Could not create PNG reader");
}

extern "C" VS_PLUGINS_EXPORT bool handleReaderFn(VideoStitch::Ptv::Value const* config) {
  return VideoStitch::Input::PngReader::handles(config);
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Input::ProbeResult probeReaderFn(std::string const& p_filename) {
  return VideoStitch::Input::PngReader::probe(p_filename);
}
//\}

/** \name Services for writer plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Output::Output>* createWriterFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSWriterPlugin::Config run_time) {
  VideoStitch::Output::PngWriter* pngWriter = VideoStitch::Output::PngWriter::create(config, run_time);
  if (pngWriter) {
    return new VideoStitch::Potential<VideoStitch::Output::Output>(pngWriter);
  }
  return new VideoStitch::Potential<VideoStitch::Output::Output>(
      VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, "Could not create PNG writer");
}

extern "C" VS_PLUGINS_EXPORT bool handleWriterFn(VideoStitch::Ptv::Value const* config) {
  return VideoStitch::Output::PngWriter::handles(config);
}
//\}
