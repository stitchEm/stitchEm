// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "resizepanoramawidget.hpp"

#include "libvideostitch-gui/utils/panoutilities.hpp"

ResizePanoramaWidget::ResizePanoramaWidget(const int width, const int height, QWidget *parent) : QWidget(parent) {
  setupUi(this);
  stackedWidget->setCurrentIndex(0);
  labelMessage->setText(labelMessage->text().arg(width).arg(height));

  // Calculate new proper dimensions
  PanoSizeSelector::PanoramaSizePreset preset = PanoSizeSelector::getPresetFromSize(width, height);
  if (int(preset) < int(PanoSizeSelector::PanoramaSizePreset::Unknown) - 1) {
    PanoSizeSelector::PanoramaSizePreset newPreset = PanoSizeSelector::PanoramaSizePreset(int(preset) + 1);
    QSize size = PanoSizeSelector::getPanoPresetSize(newPreset);
    panoSizeSelector->setSize(size.width(), size.height());
  } else {
    int newHeight = height / 2;
    const VideoStitch::Util::PanoSize size = VideoStitch::Util::calculateSizeFromHeight(newHeight);
    panoSizeSelector->setSize(size.width, size.height);
  }
}

ResizePanoramaWidget::~ResizePanoramaWidget() {}
