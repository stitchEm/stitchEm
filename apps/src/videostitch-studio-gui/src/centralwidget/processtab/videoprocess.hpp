// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QWidget>
#include "iprocesswidget.hpp"

namespace Ui {
class VideoProcess;
}

class Format;
class Codec;
class QSpinBox;

/**
 * @brief Contains all the output video configurations (encoding, format, size, etc)
 */
class VideoProcess : public IProcessWidget {
  Q_OBJECT

 public:
  explicit VideoProcess(QWidget* const parent = nullptr);
  ~VideoProcess();

 public slots:
  void onCodecConfigChanged();

 signals:
  void reqChangeFormat(const QString format);

 protected:
  virtual void reactToChangedProject() override;

 private slots:
  void updateFormatComboBox(bool hasImagesOrProceduralsOnly);
  void onCodecSelected(int index);
  void onFormatSelected(int index);

 private:
  QScopedPointer<Ui::VideoProcess> ui;
  QScopedPointer<Format> currentFormat;
  void updateCodecConfiguration(const QString codecName);
  void updateSupportedCodecs();
};
