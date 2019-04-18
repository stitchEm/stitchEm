// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/config.hpp"

#include <QDialog>

namespace Ui {
class JumpDialog;
}

/**
 * @brief Dialogbox used to jump to a give frame
 */
class JumpDialog : public QDialog {
  Q_OBJECT

 public:
  explicit JumpDialog(const frameid_t firstFrame, const frameid_t lastFrame, QWidget* const parent = nullptr);
  ~JumpDialog();
 signals:
  /**
   * @brief Sends a signal to seek to the frame given in parameter
   * @param frameToSeekTo Frame to seek to.
   */
  void reqSeek(frameid_t frameToSeekTo);

 private slots:
  /**
   * @brief Slot called when the user clicks on the button to accept.
   */
  void onButtonAcceptClicked();

 private:
  Ui::JumpDialog* ui;
};
