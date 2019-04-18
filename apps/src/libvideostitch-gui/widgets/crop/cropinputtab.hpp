// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/common.hpp"
#include "libvideostitch-gui/widgets/singlevideowidget.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include <QFrame>

namespace Ui {
class CropInputTab;
}

class VS_GUI_EXPORT CropInputTab : public QFrame {
  Q_OBJECT

 public:
  explicit CropInputTab(const int id, const QSize videoSize, const int availableHaight, const Crop& crop,
                        const InputLensClass::LensType t, QWidget* parent = nullptr);
  ~CropInputTab();

  QString getReaderName() const;
  Crop getCrop() const;
  void disableCropActions(const bool block);
  std::shared_ptr<SingleVideoWidget> getVideoWidget() const;
  void setDefaultCrop();

 public slots:
  void setCrop(const Crop& crop);

 signals:
  void cropChanged(const Crop& crop);
  /**
   * @brief Signal triggered when a new change on the crop shape is performed
   * @param crop A new crop value
   */
  void notifyCropSet(const Crop& crop);

 private:
  void createCropEditor(const QSize thumbnailSize, const Crop& crop, InputLensClass::LensType lensType);
  QSize calculateThumbnailSize() const;
  void updateEditorFromSpinBoxes();

 private slots:
  void onCropShapeChanged(const Crop& crop);
  void onLeftCropValueChanged();
  void onRightCropValueChanged();
  void onTopCropValueChanged();
  void onBottomCropValueChanged();

 private:
  Ui::CropInputTab* ui;
  const int inputIndex;
  const int maxHeight;
  const QSize videoSize;
  std::shared_ptr<SingleVideoWidget> videoWidget;
  QScopedPointer<CropShapeEditor> editor;
};
