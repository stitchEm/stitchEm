// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "stylablewidget.hpp"
#include "libvideostitch/config.hpp"

namespace Ui {
class TimeWidget;
}

class QKeyEvent;

class VS_GUI_EXPORT TimeWidget : public QWidget {
  Q_PROPERTY(bool timecodeVisibility READ getTimecodeVisibility WRITE setTimecodeVisibility DESIGNABLE true)
  Q_PROPERTY(bool framenumberVisibility READ getFramenumberVisibility WRITE setFramenumberVisibility DESIGNABLE true)
  Q_OBJECT

 public:
  explicit TimeWidget(QWidget *parent = nullptr);
  ~TimeWidget();

  /**
   * @brief Sets the timecode editable/ not editable.
   * @param True = the use can edit the timecode, false, the edits are readOnly.
   */
  void setEditable(bool editable);
  /**
   * @brief Gets the editable status.
   * @return True, the user can edit the timecode, false, the edits are readOnly.
   */
  bool isEditable() const;
  /**
   * @brief Resets the timecode to an empty string.
   */
  void reset();

  /**
   * @brief Updates the input mask depending on the input time format.
   * @param moreThanOneHour True if the format contains hour value.
   * @param threeDigitsFps True if the framerate is over 99 fps.
   */
  void updateInputMask(bool moreThanOneHour, bool threeDigitsFps);

  /**
   * @brief Sets the maximum frame that can be displayed.
   * @param max Maximum frame.
   */
  void setMaxFrame(const frameid_t max);

  /**
   * Properties
   */
  bool getTimecodeVisibility() const;
  bool getFramenumberVisibility() const;
  void setTimecodeVisibility(bool visible);
  void setFramenumberVisibility(bool visible);

 public slots:
  /**
   * @brief Sets the value of the timecode and frame to the given frame. The framerate is used to compute the timecode.
   * @param frame Frame used to set the timecode/frame.
   */
  void setFrame(frameid_t frameId);

 signals:
  /**
   * @brief Signal emitted when the frame number has been changed.
   * @param frameId New frame.
   */
  void frameChanged(frameid_t frameId);
  /**
   * @brief Signal emitted when the timecode has been changed.
   * @param frameId New frame.
   */
  void frameChangedFromTimeCode(frameid_t frameId);

 private:
  /**
   * @brief Overloaded mouse double click event to toggle on/off the editable status.
   * @param e Mouse Event.
   */
  virtual void mouseDoubleClickEvent(QMouseEvent *e) override;
  /**
   * @brief Overloaded keyboard event. May be used to apply the changes when the user will hit enter.
   * @param e Keyboard event.
   */
  virtual void keyPressEvent(QKeyEvent *e) override;

  Ui::TimeWidget *ui;
  bool editable;
  frameid_t maxFrame;
  frameid_t defaultFrame;
};
