// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef VIGNETTEVIEW_HPP
#define VIGNETTEVIEW_HPP

#include <QWidget>

class VS_GUI_EXPORT VignetteView : public QWidget {
  Q_OBJECT

 public:
  explicit VignetteView(QWidget *parent = 0);
  void paintEvent(QPaintEvent *event);

  void setVignetteParams(double vC1, double vC2, double vC3) {
    vigCoeff1 = vC1;
    vigCoeff2 = vC2;
    vigCoeff3 = vC3;
    update();
  }

  void setRenderPreview(bool renderPreview);

 private:
  void paintPreview();
  void paintGraph();

  double getVignette(float x, float y, float size);
  double vigCoeff1, vigCoeff2, vigCoeff3;
  bool shouldPreview;
};

#endif  // VIGNETTEVIEW_HPP
