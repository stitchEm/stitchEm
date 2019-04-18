// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef RESIZEPANORAMADIALOG_HPP
#define RESIZEPANORAMADIALOG_HPP

#include "generic/genericdialog.hpp"

class ResizePanoramaWidget;
class ResizePanoramaDialog : public GenericDialog {
  Q_OBJECT
 public:
  explicit ResizePanoramaDialog(const unsigned width, const unsigned height, QWidget* const parent = nullptr);

 private:
  ResizePanoramaWidget* resetWidget;

 private slots:
  void onAcceptClicked();

  void onNextClicked();

  void onCancelClicked();

 signals:
  void notifyPanoValuesSet(const int width, const int height);
};

#endif  // RESIZEPANORAMADIALOG_HPP
