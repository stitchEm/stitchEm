// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

/**
 * @brief Interface for widget that are stateful regarding the GUI.
 */
class GUIStateCaps {
 public:
  /**
   * @brief List of states.
   */
  enum State { disabled, idle, stitch, frozen };

 public:
  /**
   * @brief Changes the widget's state to the given state.
   * @param s State you want to switch to.
   */
  virtual void changeState(State s) = 0;
  virtual void reqChangeState(GUIStateCaps::State s) = 0;
};
