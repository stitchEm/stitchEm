// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "export.hpp"
#include "jpgInput.hpp"
#include "jpgOutput.hpp"
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
  VideoStitch::Input::Reader* l_return = 0;
  std::string const* filename = VideoStitch::Input::MultiFileReader::hasStringContent(config);
  if (filename) {
    l_return = VideoStitch::Input::JpgReader::create(*filename, runtime);
  }
  if (l_return) {
    return new VideoStitch::Potential<VideoStitch::Input::Reader>(l_return);
  }
  return new VideoStitch::Potential<VideoStitch::Input::Reader>(
      VideoStitch::Origin::Input, VideoStitch::ErrType::InvalidConfiguration, "Could not create Jpeg reader");
}

extern "C" VS_PLUGINS_EXPORT bool handleReaderFn(VideoStitch::Ptv::Value const* config) {
  return VideoStitch::Input::JpgReader::handles(config);
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Input::ProbeResult probeReaderFn(std::string const& p_filename) {
  return VideoStitch::Input::JpgReader::probe(p_filename);
}
//\}

/** \name Services for writer plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Output::Output>* createWriterFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSWriterPlugin::Config run_time) {
  VideoStitch::Potential<VideoStitch::Output::JpgWriter> jpgWriter =
      VideoStitch::Output::JpgWriter::create(config, run_time);
  if (jpgWriter.ok()) {
    return new VideoStitch::Potential<VideoStitch::Output::Output>(jpgWriter.release());
  }
  return new VideoStitch::Potential<VideoStitch::Output::Output>(jpgWriter.status());
}

extern "C" VS_PLUGINS_EXPORT bool handleWriterFn(VideoStitch::Ptv::Value const* config) {
  return VideoStitch::Output::JpgWriter::handles(config);
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
