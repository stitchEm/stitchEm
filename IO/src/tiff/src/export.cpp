// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "export.hpp"
#include "tiffOutput.hpp"

#include "libgpudiscovery/delayLoad.hpp"

#include "libvideostitch/output.hpp"
#include "libvideostitch/plugin.hpp"

#ifdef DELAY_LOAD_ENABLED
SET_DELAY_LOAD_HOOK
#endif  // DELAY_LOAD_ENABLED

/** \name Services for writer plugin. */
//\{
extern "C" VS_PLUGINS_EXPORT VideoStitch::Potential<VideoStitch::Output::Output>* createWriterFn(
    VideoStitch::Ptv::Value const* config, VideoStitch::Plugin::VSWriterPlugin::Config run_time) {
  VideoStitch::Output::BaseConfig baseConfig;
  const VideoStitch::Status parseStatus = baseConfig.parse(*config);
  if (parseStatus.ok()) {
    auto potTiffWriter = VideoStitch::Output::TiffWriter::create(
        *config, baseConfig.baseName, run_time.width, run_time.height, run_time.framerate,
        VideoStitch::Output::NumberedFilesWriter::readReferenceFrame(*config));
    if (potTiffWriter.ok()) {
      return new VideoStitch::Potential<VideoStitch::Output::Output>(potTiffWriter.release());
    } else {
      return new VideoStitch::Potential<VideoStitch::Output::Output>(potTiffWriter.status());
    }
  }
  return new VideoStitch::Potential<VideoStitch::Output::Output>(
      VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration,
      "Could not parse TIFF writer configuration", parseStatus);
}

extern "C" VS_PLUGINS_EXPORT bool handleWriterFn(VideoStitch::Ptv::Value const* config) {
  bool l_return = false;
  VideoStitch::Output::BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    l_return = (!strcmp(baseConfig.strFmt, "tif"));
  }
  return l_return;
}
//\}
