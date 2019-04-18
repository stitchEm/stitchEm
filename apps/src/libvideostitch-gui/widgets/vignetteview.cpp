// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "vignetteview.hpp"
#include <QPainter>
#include <QMoveEvent>

#include <iostream>

static const int vignetteViewSize = 189;

VignetteView::VignetteView(QWidget *parent)
    : QWidget(parent), vigCoeff1(0.0), vigCoeff2(0.0), vigCoeff3(0.0), shouldPreview(false) {}

double VignetteView::getVignette(float x, float y, float size) {
  double radiusSquared = x / size / 2 * x / size / 2 + y / size / 2 * y / size / 2;

  double vigMult = radiusSquared * vigCoeff3;
  vigMult += vigCoeff2;
  vigMult *= radiusSquared;
  vigMult += vigCoeff1;
  vigMult *= radiusSquared;
  vigMult += 1;  // vigCoeff0 = 1
  return vigMult;
}

void VignetteView::setRenderPreview(bool renderPreview) {
  shouldPreview = renderPreview;
  update();
}

void VignetteView::paintPreview() {
  QPainter painter(this);

  int referenceColor = 127;

  painter.setPen(QColor(referenceColor, referenceColor, referenceColor));
  painter.setBrush(QColor(referenceColor, referenceColor, referenceColor));
  painter.drawRect(0, 0, vignetteViewSize, vignetteViewSize);

  // have a small border around the plotted vignette as a reference
  int margin = 3;

  // TODO fill rect with GLSL shader or comparable
  for (int x = 0; x < vignetteViewSize - margin; ++x) {
    for (int y = 0; y < vignetteViewSize - margin; ++y) {
      double vigMult = getVignette(x, y, vignetteViewSize - margin);
      int vigColor = referenceColor * vigMult;
      vigColor = std::min(vigColor, 255);
      vigColor = std::max(vigColor, 0);
      painter.setPen(QColor(vigColor, vigColor, vigColor));
      painter.drawPoint(x, vignetteViewSize - y);
    }
  }
}

void VignetteView::paintGraph() {
  QPainter painter(this);

  painter.setPen(QColor(95, 95, 95));
  painter.drawRect(0, 0, vignetteViewSize, vignetteViewSize);

  QPointF *qpoints = new QPointF[vignetteViewSize];

  for (int i = 0; i < vignetteViewSize; ++i) {
    qpoints[i].setX(i);
    double vignette = getVignette(i, i, vignetteViewSize) * 0.8;
    vignette = std::min(vignette, 1.0);
    vignette = std::max(vignette, 0.0);
    qpoints[i].setY((1 - vignette) * vignetteViewSize);
  }

  painter.setRenderHint(QPainter::Antialiasing);
  painter.drawPolyline(qpoints, vignetteViewSize);
}

void VignetteView::paintEvent(QPaintEvent *) {
  if (shouldPreview) {
    paintPreview();
  } else {
    paintGraph();
  }
}
