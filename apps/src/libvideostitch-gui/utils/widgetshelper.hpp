// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/common.hpp"

class QWidget;

namespace VideoStitch {
namespace WidgetsHelpers {

/**
 * @brief Puts the widget in from of the other windows.
 * @param widget Element to be taked to foreground.
 */
VS_GUI_EXPORT void bringToForeground(QWidget* widget);

}  // namespace WidgetsHelpers
}  // namespace VideoStitch
