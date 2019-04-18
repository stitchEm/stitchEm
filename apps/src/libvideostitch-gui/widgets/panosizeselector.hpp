// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/utils/panoutilities.hpp"

#include <QWidget>

namespace Ui {
class PanoSizeSelector;
}

class VS_GUI_EXPORT PanoSizeSelector : public QWidget {
  Q_OBJECT

 public:
  enum class PanoramaSizePreset { Cinema4K, UltraHD, TwoK, HD, Unknown };
  static QString getPanoPresetName(PanoramaSizePreset preset);
  static QSize getPanoPresetSize(PanoramaSizePreset preset);
  static PanoramaSizePreset getPresetFromSize(int width, int height);

 public:
  explicit PanoSizeSelector(QWidget* parent = nullptr);
  ~PanoSizeSelector();

  void setSize(int width, int height);
  int getWidth() const;
  int getHeight() const;

 signals:
  void sizeChanged();

 private:
  void addPresetToComboBox(PanoramaSizePreset preset);
  void updateSizeSpinBoxes(VideoStitch::Util::PanoSize size);

 private slots:
  void updateSize();
  void onWidthChanged();
  void onHeightChanged();

 private:
  QScopedPointer<Ui::PanoSizeSelector> ui;
};
