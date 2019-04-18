// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "maintabs.hpp"

#include <QCoreApplication>

namespace GuiEnums {

QString getTabName(GuiEnums::Tab tab) {
  switch (tab) {
    case Tab::TabSources:
      return QCoreApplication::translate("Main tab name", "Sources");
    case Tab::TabOutPut:
      return QCoreApplication::translate("Main tab name", "Panorama");
    case Tab::TabInteractive:
      return QCoreApplication::translate("Main tab name", "Interactive");
    case Tab::TabConfiguration:
      return QCoreApplication::translate("Main tab name", "Configuration");
    case Tab::TabCount:
    default:
      return QString();
  }
}

QString getPixmapPath(Tab tab, bool enabled) {
  QString iconFileName = ":/live/icons/assets/icon/live/";
  switch (tab) {
    case Tab::TabSources:
      iconFileName += "icon-sources-tab";
      break;
    case Tab::TabOutPut:
      iconFileName += "icon-output-tab";
      break;
    case Tab::TabInteractive:
      iconFileName += "icon-interactive-tab";
      break;
    case Tab::TabConfiguration:
      iconFileName += "icon-config-tab";
      break;
    case Tab::TabCount:
    default:
      return QString();
  }
  if (!enabled) {
    iconFileName += "-disabled";
  }
  iconFileName += ".png";
  return iconFileName;
}

}  // namespace GuiEnums
