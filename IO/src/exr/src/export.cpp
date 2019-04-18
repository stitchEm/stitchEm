// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "export.hpp"
#include "exrInput.hpp"
#include "exrOutput.hpp"
#include "numberedFilesOutput.hpp"

#include "libgpudiscovery/delayLoad.hpp"

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/plugin.hpp"
#include "libvideostitch/ptv.hpp"
#include <ostream>

#ifdef _MSC_VER
SET_DELAY_LOAD_HOOK
#endif  // _MSC_VER

/** \name Services for reader plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Input::Reader>* createReaderFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSReaderPlugin::Config runtime) {
  VideoStitch::Input::ExrReader* exrReader = VideoStitch::Input::ExrReader::create(config, runtime);
  if (exrReader) {
    return new VideoStitch::Potential<VideoStitch::Input::Reader>(exrReader);
  }
  return new VideoStitch::Potential<VideoStitch::Input::Reader>(
      VideoStitch::Origin::Input, VideoStitch::ErrType::InvalidConfiguration, "Could not create EXR reader");
}

extern "C" VS_PLUGINS_EXPORT bool handleReaderFn(VideoStitch::Ptv::Value const* config) {
  return VideoStitch::Input::ExrReader::handles(config);
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Input::ProbeResult probeReaderFn(std::string const& p_filename) {
  return VideoStitch::Input::ExrReader::probe(p_filename);
}
//\}

/** \name Services for writer plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Output::Output>* createWriterFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSWriterPlugin::Config run_time) {
  VideoStitch::Output::ExrWriter* exrWriter = VideoStitch::Output::ExrWriter::create(config, run_time);
  if (exrWriter) {
    return new VideoStitch::Potential<VideoStitch::Output::Output>(exrWriter);
  }
  return new VideoStitch::Potential<VideoStitch::Output::Output>(
      VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, "Could not create EXR writer");
}

extern "C" VS_PLUGINS_EXPORT bool handleWriterFn(VideoStitch::Ptv::Value const* config) {
  return VideoStitch::Output::ExrWriter::handles(config);
}
//\}
