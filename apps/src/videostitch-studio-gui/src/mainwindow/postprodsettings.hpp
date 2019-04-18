// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef POSTPRODSETTINGS_HPP
#define POSTPRODSETTINGS_HPP

#include "libvideostitch-gui/mainwindow/vssettings.hpp"

class PostProdSettings : public VSSettings {
 public:
  static PostProdSettings* createPostProdSettings();  // The application need to be created
  static PostProdSettings* getPostProdSettings();

 private:
  explicit PostProdSettings(const QString settingsName);
};

#endif  // POSTPRODSETTINGS_HPP
