// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "msgboxhandlerhelper.hpp"

#include "LibLogHelpers.hpp"

namespace MsgBoxHandlerHelper {

void genericErrorMessage(const VideoStitch::Status& status) {
  return MsgBoxHandler::getInstance()->generic(
      QString::fromStdString(status.getErrorMessage()), VideoStitch::Helper::createTitle(status), CRITICAL_ERROR_ICON,
      QFlags<QMessageBox::StandardButton>(QMessageBox::Ok),
      status.hasCause() ? VideoStitch::Helper::createErrorBacktrace(status.getCause()) : QString());
}

int genericErrorMessageSync(const VideoStitch::Status& status) {
  return MsgBoxHandler::getInstance()->genericSync(
      QString::fromStdString(status.getErrorMessage()), VideoStitch::Helper::createTitle(status), CRITICAL_ERROR_ICON,
      QFlags<QMessageBox::StandardButton>(QMessageBox::Ok),
      status.hasCause() ? VideoStitch::Helper::createErrorBacktrace(status.getCause()) : QString());
}

}  // namespace MsgBoxHandlerHelper
