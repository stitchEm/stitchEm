// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "stillimagehandler.hpp"

class JpgExtensionHandler : public StillImageHandler {
 public:
  JpgExtensionHandler() {
    extension = "jpg";
    format = "jpg";
  }
};
