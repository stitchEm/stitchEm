// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef STATICOUTPUTS_HPP
#define STATICOUTPUTS_HPP

#include "libvideostitch/plugin.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"
#include <QCoreApplication>
/**
 * @brief Fake output devices for common output types
 */
static VideoStitch::Plugin::DiscoveryDevice deviceHDD;
static VideoStitch::Plugin::DiscoveryDevice deviceRTMP;
// VSA-6594
#ifdef ENABLE_YOUTUBE_OUTPUT
static VideoStitch::Plugin::DiscoveryDevice deviceYoutube;
#endif
static VideoStitch::Plugin::DiscoveryDevice deviceOculus;
static VideoStitch::Plugin::DiscoveryDevice deviceSteamVR;

/**
 * @brief Add information to output types
 */
static void loadStaticOutputs() {
  deviceHDD.name =
      VideoStitch::OutputFormat::getStringFromEnum(VideoStitch::OutputFormat::OutputFormatEnum::MP4).toStdString();
  deviceRTMP.name =
      VideoStitch::OutputFormat::getStringFromEnum(VideoStitch::OutputFormat::OutputFormatEnum::RTMP).toStdString();
  deviceOculus.name =
      VideoStitch::OutputFormat::getStringFromEnum(VideoStitch::OutputFormat::OutputFormatEnum::OCULUS).toStdString();
  deviceSteamVR.name =
      VideoStitch::OutputFormat::getStringFromEnum(VideoStitch::OutputFormat::OutputFormatEnum::STEAMVR).toStdString();
  // VSA-6594
#ifdef ENABLE_YOUTUBE_OUTPUT
  deviceYoutube.name =
      VideoStitch::OutputFormat::getStringFromEnum(VideoStitch::OutputFormat::OutputFormatEnum::YOUTUBE).toStdString();
#endif
  deviceHDD.displayName = QWidget::tr("HDD file output").toStdString();
  deviceRTMP.displayName = QWidget::tr("RTMP stream").toStdString();
  deviceOculus.displayName = QWidget::tr("Oculus view").toStdString();
  deviceSteamVR.displayName = QWidget::tr("SteamVR view").toStdString();
  // VSA-6594
#ifdef ENABLE_YOUTUBE_OUTPUT
  deviceYoutube.displayName = QWidget::tr("YouTube").toStdString();
#endif
}

#endif  // STATICOUTPUTS_HPP
