// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveoutputcustom.hpp"
#include "liveprojectdefinition.hpp"

#include "libvideostitch-base/logmanager.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"
#include <QLabel>

LiveOutputCustom::LiveOutputCustom(const VideoStitch::Ptv::Value* config,
                                   const VideoStitch::OutputFormat::OutputFormatEnum outputType)
    : LiveOutputFactory(outputType), config(config->clone()) {
  std::string name;
  if (VideoStitch::Parse::populateString("Ptv", *config, "type", name, true) == VideoStitch::Parse::PopulateResult_Ok) {
    customType = QString::fromStdString(name);
  }
}

const QString LiveOutputCustom::getIdentifier() const { return QStringLiteral("custom"); }

const QString LiveOutputCustom::getOutputTypeDisplayName() const { return customType; }

const QString LiveOutputCustom::getOutputDisplayName() const { return QString(); }

VideoStitch::Ptv::Value* LiveOutputCustom::serialize() const { return config->clone(); }

QWidget* LiveOutputCustom::createStatusWidget(QWidget* const parent) { return createStatusIcon(parent); }

QPixmap LiveOutputCustom::getIcon() const { return QPixmap(":/live/icons/assets/icon/live/output.png"); }
