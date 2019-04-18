// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef DIALOGBACKGROUND_HPP
#define DIALOGBACKGROUND_HPP
#include <QFrame>

class DialogBackground : public QFrame {
  Q_OBJECT

 public:
  explicit DialogBackground(QWidget* const parent = nullptr);

  void updateSize(unsigned int parentWidth, unsigned int parentHeight);
};

#endif  // DIALOGBACKGROUND_HPP
