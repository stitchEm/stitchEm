// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PHOTOMETRICCALIBRATIONWIDGET_HPP
#define PHOTOMETRICCALIBRATIONWIDGET_HPP

#include <QWidget>

namespace Ui {
class PhotometricCalibrationWidget;
}

class PhotometricCalibrationWidget : public QWidget {
  Q_OBJECT
 public:
  explicit PhotometricCalibrationWidget(QWidget* const parent = nullptr);
  ~PhotometricCalibrationWidget();

  void setEmorValues(double emor1, double emor2, double emor3, double emor4, double emor5);
  void setVignetteValues(double vC1, double vC2, double vC3);

 private slots:
  void onRenderVignetteClicked();

 private:
  QScopedPointer<Ui::PhotometricCalibrationWidget> ui;
  bool renderVignette;
};

#endif  // PHOTOMETRICCALIBRATIONWIDGET_HPP
