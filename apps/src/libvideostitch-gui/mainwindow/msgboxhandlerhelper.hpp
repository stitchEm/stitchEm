// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/common.hpp"

#include "libvideostitch-base/msgboxhandler.hpp"

#include "libvideostitch/status.hpp"

namespace MsgBoxHandlerHelper {
/**
 * @brief generic requests a generic message box. Thread-safe.
 * @param status Status object containing error information to be displayed
 */
void VS_GUI_EXPORT genericErrorMessage(const VideoStitch::Status& status);

/**
 * @brief generic requests a message box with a return value. Not thread-safe, must be executed from the GUI thread.
 * @param status Status object containing error information to be displayed
 * @return the QMessageBox return value.
 */
int VS_GUI_EXPORT genericErrorMessageSync(const VideoStitch::Status& status);
}  // namespace MsgBoxHandlerHelper
