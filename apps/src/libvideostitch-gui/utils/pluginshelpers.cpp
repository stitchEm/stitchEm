// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "pluginshelpers.hpp"

#include "libvideostitch/gpu_device.hpp"

#include <QCoreApplication>
#include <QDir>

namespace VideoStitch {
namespace Plugin {

QString getCorePluginFolderPath() {
#if defined(__APPLE__)
  return QCoreApplication::applicationDirPath() + QDir::separator() + "../CorePlugins";
#else
  return QCoreApplication::applicationDirPath() + QDir::separator() + "core_plugins";
#endif
}

QString getGpuCorePluginFolderPath() {
#if defined(_MSC_VER)
  return QCoreApplication::applicationDirPath() + QDir::separator() + "core_plugins_" +
         QString::fromStdString(VideoStitch::Discovery::getFrameworkName(VideoStitch::GPU::getFramework()));
#else
  return QString();
#endif
}

}  // namespace Plugin
}  // namespace VideoStitch
