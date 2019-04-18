// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

/**
 * @brief Interface for every widget contained in a QDockWidget accesible by menu Window.
 */
class IToolWidget {
 public:
  IToolWidget() {}

  /**
   * @brief Resets the widget.
   */
  virtual void reset() = 0;
};
