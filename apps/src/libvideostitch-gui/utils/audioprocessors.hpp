// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QString>
#include <QApplication>

namespace VideoStitch {
namespace AudioProcessors {

enum class ProcessorEnum { UNKNOWN, DELAY };

static inline QString getDisplayNameFromEnum(const ProcessorEnum& value) {
  switch (value) {
    case ProcessorEnum::DELAY:
      return QApplication::translate("Processor", "Audio delay");
    default:
      return QString();
  }
}

static inline QString getStringFromEnum(const ProcessorEnum& value) {
  switch (value) {
    case ProcessorEnum::DELAY:
      return QStringLiteral("delay");
    default:
      return QString();
  }
}

static inline ProcessorEnum getEnumFromString(const QString& value) {
  if (value == "delay") {
    return ProcessorEnum::DELAY;
  } else {
    return ProcessorEnum::UNKNOWN;
  }
}

}  // namespace AudioProcessors
}  // namespace VideoStitch
