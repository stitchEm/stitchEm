// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QWidget>
/**
 * @brief An interface for every Studio tab widget.
 */
class VS_GUI_EXPORT ICentralTabWidget {
 public:
  /**
   * @brief Playback feature can be performed in this tab
   * @return True if allows playback.
   */
  virtual bool allowsPlayback() const { return false; }
};
