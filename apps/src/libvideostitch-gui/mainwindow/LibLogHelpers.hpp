// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/status.hpp"

#include <QString>
#include <QHash>

namespace VideoStitch {
namespace Helper {

VS_GUI_EXPORT QString createTitle(const VideoStitch::Status& status);
VS_GUI_EXPORT QString createErrorBacktrace(const VideoStitch::Status& status);

}  // namespace Helper
}  // namespace VideoStitch
