// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panosizeselector.hpp"
#include "ui_panosizeselector.h"

QString PanoSizeSelector::getPanoPresetName(PanoSizeSelector::PanoramaSizePreset preset) {
  switch (preset) {
    case PanoramaSizePreset::Cinema4K:
      return QStringLiteral("Cinema 4K");
    case PanoramaSizePreset::UltraHD:
      return QStringLiteral("Ultra HD");
    case PanoramaSizePreset::TwoK:
      return QStringLiteral("2K");
    case PanoramaSizePreset::HD:
      return QStringLiteral("HD");
    case PanoramaSizePreset::Unknown:
      return QString();
  }
  return QString();
}

QSize PanoSizeSelector::getPanoPresetSize(PanoSizeSelector::PanoramaSizePreset preset) {
  switch (preset) {
    case PanoramaSizePreset::Cinema4K:
      return QSize(4096, 2048);
    case PanoramaSizePreset::UltraHD:
      return QSize(3840, 1920);
    case PanoramaSizePreset::TwoK:
      return QSize(2048, 1024);
    case PanoramaSizePreset::HD:
      return QSize(1920, 960);
    case PanoramaSizePreset::Unknown:
      return QSize();
  }
  return QSize();
}

PanoSizeSelector::PanoramaSizePreset PanoSizeSelector::getPresetFromSize(int width, int height) {
  for (int preset = int(PanoramaSizePreset::Cinema4K); preset != int(PanoramaSizePreset::Unknown); ++preset) {
    QSize size = getPanoPresetSize(PanoramaSizePreset(preset));
    if (size.width() == width && size.height() == height) {
      return PanoramaSizePreset(preset);
    }
  }
  return PanoramaSizePreset::Unknown;
}

PanoSizeSelector::PanoSizeSelector(QWidget *parent) : QWidget(parent), ui(new Ui::PanoSizeSelector) {
  ui->setupUi(this);
  connect(ui->spinWidth, &QSpinBox::editingFinished, this, &PanoSizeSelector::onWidthChanged);
  connect(ui->spinHeight, &QSpinBox::editingFinished, this, &PanoSizeSelector::onHeightChanged);
  connect(ui->presetBox, &QComboBox::currentTextChanged, this, &PanoSizeSelector::updateSize);

  addPresetToComboBox(PanoramaSizePreset::Cinema4K);
  addPresetToComboBox(PanoramaSizePreset::UltraHD);
  addPresetToComboBox(PanoramaSizePreset::TwoK);
  addPresetToComboBox(PanoramaSizePreset::HD);
  //: Custom panorama size
  ui->presetBox->addItem(tr("Custom"), int(PanoramaSizePreset::Unknown));
  updateSize();
}

PanoSizeSelector::~PanoSizeSelector() {}

void PanoSizeSelector::setSize(int width, int height) {
  const PanoramaSizePreset foundPreset = getPresetFromSize(width, height);
  ui->presetBox->setCurrentIndex(ui->presetBox->findData(int(foundPreset)));
  if (foundPreset == PanoramaSizePreset::Unknown) {
    ui->spinWidth->setValue(width);
    ui->spinHeight->setValue(height);
  }
}

int PanoSizeSelector::getWidth() const { return ui->spinWidth->value(); }

int PanoSizeSelector::getHeight() const { return ui->spinHeight->value(); }

void PanoSizeSelector::addPresetToComboBox(PanoSizeSelector::PanoramaSizePreset preset) {
  const QSize size = getPanoPresetSize(preset);
  const QString text = QString("%0 (%1x%2)").arg(getPanoPresetName(preset)).arg(size.width()).arg(size.height());
  ui->presetBox->addItem(text, int(preset));
}

void PanoSizeSelector::updateSizeSpinBoxes(VideoStitch::Util::PanoSize size) {
  if (ui->spinWidth->value() != size.width) {
    ui->spinWidth->setValue(size.width);
  }
  if (ui->spinHeight->value() != size.height) {
    ui->spinHeight->setValue(size.height);
  }
  emit sizeChanged();
}

void PanoSizeSelector::updateSize() {
  const PanoSizeSelector::PanoramaSizePreset preset =
      PanoSizeSelector::PanoramaSizePreset(ui->presetBox->currentData().toInt());
  ui->spinWidth->setVisible(preset == PanoramaSizePreset::Unknown);
  ui->spinHeight->setVisible(preset == PanoramaSizePreset::Unknown);
  ui->labelX->setVisible(preset == PanoramaSizePreset::Unknown);
  if (preset != PanoramaSizePreset::Unknown) {
    QSize size = getPanoPresetSize(preset);
    ui->spinWidth->setValue(size.width());
    ui->spinHeight->setValue(size.height());
  }
  emit sizeChanged();
}

void PanoSizeSelector::onWidthChanged() {
  const VideoStitch::Util::PanoSize size = VideoStitch::Util::calculateSizeFromWidth(ui->spinWidth->value());
  updateSizeSpinBoxes(size);
}

void PanoSizeSelector::onHeightChanged() {
  const VideoStitch::Util::PanoSize size = VideoStitch::Util::calculateSizeFromHeight(ui->spinHeight->value());
  updateSizeSpinBoxes(size);
}
