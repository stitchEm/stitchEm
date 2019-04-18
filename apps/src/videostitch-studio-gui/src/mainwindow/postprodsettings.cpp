// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "postprodsettings.hpp"

#include "libvideostitch-base/common-config.hpp"

PostProdSettings::PostProdSettings(const QString settingsName) : VSSettings(settingsName) {}

PostProdSettings* PostProdSettings::createPostProdSettings() {
  Q_ASSERT(qApp != nullptr);
  PostProdSettings* postProdSettings = getPostProdSettings();
  if (!postProdSettings) {
    postProdSettings = new PostProdSettings(VIDEOSTITCH_STUDIO_SETTINGS_NAME);
    postProdSettings->setParent(qApp);
  }
  return postProdSettings;
}

PostProdSettings* PostProdSettings::getPostProdSettings() {
  Q_ASSERT(qApp != nullptr);
  return qApp->findChild<PostProdSettings*>();
}
