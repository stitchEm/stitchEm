// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef AUTOSELECTSPINBOX_HPP
#define AUTOSELECTSPINBOX_HPP

#include <QSpinBox>
#include <QFocusEvent>
/**
 * @brief The AutoSelectSpinBox class is used to display a sinpbox that autoselects its content when focused.
 */
class VS_GUI_EXPORT AutoSelectSpinBox : public QSpinBox {
  Q_OBJECT
 public:
  explicit AutoSelectSpinBox(QWidget *parent = nullptr) : QSpinBox(parent), autoSelectOnFocus(false) {
    setKeyboardTracking(false);
    setWrapping(false);
    setAccelerated(true);
  }
  /**
   * @brief setAutoSelectOnFocus Turns on/off the auto select feature.
   * @param autoSelectOnFocus
   */
  void setAutoSelectOnFocus(bool autoSelectOnFocus) { this->autoSelectOnFocus = autoSelectOnFocus; }

 protected:
  /**
   * @brief focusInEvent Overloaded focus event to select its content when the feature is turned on.
   * @param event
   */
  void focusInEvent(QFocusEvent *event) {
    if (autoSelectOnFocus) {
      selectAll();
    }
    event->accept();
    QSpinBox::focusInEvent(event);
  }

 private:
  bool autoSelectOnFocus;
};

#endif  // AUTOSELECTSPINBOX_HPP
