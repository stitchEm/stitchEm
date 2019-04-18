// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "export.hpp"
#include "videoReader.hpp"
#include "netStreamReader.hpp"
#include "avWriter.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/plugin.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/status.hpp"

#include <ostream>

/** \name Services for reader plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Input::Reader>* createReaderFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSReaderPlugin::Config runtime) {
  if (VideoStitch::Input::FFmpegReader::handles(config->asString()) ||
      VideoStitch::Input::netStreamReader::handles(config->asString())) {
    auto potLibAvReader = VideoStitch::Input::LibavReader::create(config->asString(), runtime);
    if (potLibAvReader.ok()) {
      return new VideoStitch::Potential<VideoStitch::Input::Reader>(potLibAvReader.release());
    } else {
      return new VideoStitch::Potential<VideoStitch::Input::Reader>(potLibAvReader.status());
    }
  }
  return new VideoStitch::Potential<VideoStitch::Input::Reader>{VideoStitch::Origin::Input,
                                                                VideoStitch::ErrType::InvalidConfiguration,
                                                                "Reader doesn't handle this configuration"};
}

extern "C" VS_PLUGINS_EXPORT bool handleReaderFn(VideoStitch::Ptv::Value const* config) {
  if (config && config->getType() == VideoStitch::Ptv::Value::STRING) {
    return (VideoStitch::Input::FFmpegReader::handles(config->asString()) ||
            VideoStitch::Input::netStreamReader::handles(config->asString()));
  } else {
    return false;
  }
}

extern "C" VS_PLUGINS_EXPORT VideoStitch::Input::ProbeResult probeReaderFn(std::string const& p_filename) {
  return VideoStitch::Input::LibavReader::probe(p_filename);
}
//\}

/** \name Services for writer plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Output::Output>* createWriterFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSWriterPlugin::Config run_time) {
  VideoStitch::Output::Output* lReturn = nullptr;
  VideoStitch::Output::BaseConfig baseConfig;
  const VideoStitch::Status parseStatus = baseConfig.parse(*config);
  if (parseStatus.ok()) {
    lReturn = VideoStitch::Output::LibavWriter::create(*config, run_time.name, baseConfig.baseName, run_time.width,
                                                       run_time.height, run_time.framerate, run_time.rate,
                                                       run_time.depth, run_time.layout);
    if (lReturn) {
      return new VideoStitch::Potential<VideoStitch::Output::Output>(lReturn);
    }
    return new VideoStitch::Potential<VideoStitch::Output::Output>(
        VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, "Could not create av writer");
  }
  return new VideoStitch::Potential<VideoStitch::Output::Output>(
      VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration,
      "Could not parse AV Writer configuration", parseStatus);
}

extern "C" VS_PLUGINS_EXPORT bool handleWriterFn(VideoStitch::Ptv::Value const* config) {
  bool lReturn(false);
  VideoStitch::Output::BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    lReturn = (!strcmp(baseConfig.strFmt, "mp4") || !strcmp(baseConfig.strFmt, "mov"));
  } else {
    // TODOLATERSTATUS
    VideoStitch::Logger::get(VideoStitch::Logger::Verbose) << "avPlugin: cannot parse BaseConfnig" << std::endl;
  }
  return lReturn;
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
