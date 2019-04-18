// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "widgetshelper.hpp"
#include <QWidget>

#ifdef Q_OS_WIN
#include <Windows.h>
#endif

namespace VideoStitch {
namespace WidgetsHelpers {

void bringToForeground(QWidget* widget) {
  widget->activateWindow();
  widget->raise();
  widget->showMaximized();
  // WIN32 stuff to bring back the window to foreground.
#ifdef Q_OS_WIN
  HWND currentWindow = (HWND)widget->effectiveWinId();
  SetForegroundWindow(currentWindow);
  SetActiveWindow(currentWindow);
#endif
}

}  // namespace WidgetsHelpers
}  // namespace VideoStitch
