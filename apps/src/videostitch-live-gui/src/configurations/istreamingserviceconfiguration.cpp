// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "istreamingserviceconfiguration.hpp"
#include "basertmpserviceconfiguration.hpp"
#ifdef ENABLE_YOUTUBE_OUTPUT
#include "youtubebaseconfiguration.hpp"
#endif
#include "libvideostitch-gui/utils/outputformat.hpp"
#include "videostitcher/liveoutputrtmp.hpp"

#include <unordered_map>
#include <functional>

IStreamingServiceConfiguration::IStreamingServiceConfiguration(QWidget* parent, LiveOutputRTMP* outputref,
                                                               LiveProjectDefinition* projectDefinition)
    : QGroupBox(parent), outputRef(outputref), liveProjectDefinition(projectDefinition) {}

IStreamingServiceConfiguration* IStreamingServiceConfiguration::createStreamingService(
    QWidget* parent, LiveOutputRTMP* outputRef, LiveProjectDefinition* projectDefinition) {
  if (outputRef->getType() == VideoStitch::OutputFormat::OutputFormatEnum::RTMP) {
    return new BaseRTMPServiceConfiguration(parent, outputRef, projectDefinition);
#ifdef ENABLE_YOUTUBE_OUTPUT
  } else if (outputRef->getType() == VideoStitch::OutputFormat::OutputFormatEnum::YOUTUBE) {
    return new YoutubeBaseConfiguration(parent, outputRef, projectDefinition);
#endif
  } else {
    return nullptr;
  }
}

void IStreamingServiceConfiguration::setLiveProjectDefinition(LiveProjectDefinition* value) {
  liveProjectDefinition = value;
}
