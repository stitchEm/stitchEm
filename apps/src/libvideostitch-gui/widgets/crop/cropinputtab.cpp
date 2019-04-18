// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "cropinputtab.hpp"
#include "ui_cropinputtab.h"
#include "libvideostitch-gui/utils/inputlensenum.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"
#include "libvideostitch-gui/widgets/crop/croprectangleeditor.hpp"
#include "libvideostitch-gui/widgets/crop/cropcircleeditor.hpp"

CropInputTab::CropInputTab(const int id, const QSize videoSize, const int availableHaight, const Crop& crop,
                           const InputLensClass::LensType t, QWidget* parent)
    : QFrame(parent),
      ui(new Ui::CropInputTab),
      inputIndex(id),
      maxHeight(availableHaight),
      videoSize(videoSize),
      videoWidget(nullptr),
      editor(nullptr) {
  ui->setupUi(this);
  ui->buttonRestore->setProperty("vs-button-medium", true);
  videoWidget = std::shared_ptr<SingleVideoWidget>(new SingleVideoWidget(id, ui->frameBackground));
  videoWidget->setObjectName(QStringLiteral("videoWidget"));
  videoWidget->setMinimumSize(QSize(0, 0));
  ui->layoutMain->addWidget(videoWidget.get());
  videoWidget->setName(getReaderName().toStdString());
  connect(ui->spinLeft, &QSpinBox::editingFinished, this, &CropInputTab::onLeftCropValueChanged);
  connect(ui->spinTop, &QSpinBox::editingFinished, this, &CropInputTab::onTopCropValueChanged);
  connect(ui->spinRight, &QSpinBox::editingFinished, this, &CropInputTab::onRightCropValueChanged);
  connect(ui->spinBottom, &QSpinBox::editingFinished, this, &CropInputTab::onBottomCropValueChanged);
  createCropEditor(calculateThumbnailSize(), crop, t);
  videoWidget->setFixedSize(calculateThumbnailSize());
}

CropInputTab::~CropInputTab() { delete ui; }

QString CropInputTab::getReaderName() const { return QStringLiteral("cropInputTab") + QString::number(inputIndex); }

Crop CropInputTab::getCrop() const { return editor->getCrop(); }

void CropInputTab::disableCropActions(const bool block) {
  ui->topFrame->setDisabled(block);
  editor->disableEdition(block);
}

std::shared_ptr<SingleVideoWidget> CropInputTab::getVideoWidget() const { return videoWidget; }

void CropInputTab::setDefaultCrop() {
  if (editor != nullptr) {
    editor->setDefaultCrop();
  }
}

void CropInputTab::updateEditorFromSpinBoxes() {
  editor->setCrop(Crop(ui->spinLeft->value(), ui->spinRight->value(), ui->spinTop->value(), ui->spinBottom->value()));
}

void CropInputTab::onCropShapeChanged(const Crop& crop) {
  ui->spinLeft->setValue(crop.crop_left);
  ui->spinTop->setValue(crop.crop_top);
  ui->spinRight->setValue(crop.crop_right);
  ui->spinBottom->setValue(crop.crop_bottom);
  emit cropChanged(crop);
}

void CropInputTab::setCrop(const Crop& crop) { editor->setCrop(crop); }

void CropInputTab::createCropEditor(const QSize thumbnailSize, const Crop& crop, InputLensClass::LensType lensType) {
  if (lensType == InputLensClass::LensType::CircularFisheye) {
    editor.reset(new CropCircleEditor(thumbnailSize, videoSize, crop, videoWidget.get()));
  } else {
    editor.reset(new CropRectangleEditor(thumbnailSize, videoSize, crop, videoWidget.get()));
  }
  connect(editor.data(), &CropShapeEditor::notifyCropSet, this, &CropInputTab::onCropShapeChanged);
  connect(ui->buttonRestore, &QPushButton::clicked, editor.data(), &CropShapeEditor::onResetToDefault);
  onCropShapeChanged(editor->getCrop());
}

QSize CropInputTab::calculateThumbnailSize() const {
  if (maxHeight < (videoSize.height())) {
    // If it doesn't fit in the available space, reduce it
    const float factor = floor(float(maxHeight) / float(videoSize.height()) * 10 + .5) / 10;
    return QSize(videoSize.width() * factor, videoSize.height() * factor);
  } else {
    // Otherwise, use the original frame size
    return videoSize;
  }
}

void CropInputTab::onLeftCropValueChanged() {
  if (ui->spinLeft->value() > ui->spinRight->value()) {
    ui->spinLeft->blockSignals(true);
    ui->spinLeft->setValue(ui->spinRight->value());
    ui->spinLeft->blockSignals(false);
  }
  updateEditorFromSpinBoxes();
}

void CropInputTab::onRightCropValueChanged() {
  if (ui->spinLeft->value() > ui->spinRight->value()) {
    ui->spinRight->blockSignals(true);
    ui->spinRight->setValue(ui->spinLeft->value());
    ui->spinRight->blockSignals(false);
  }
  updateEditorFromSpinBoxes();
}

void CropInputTab::onTopCropValueChanged() {
  if (ui->spinTop->value() > ui->spinBottom->value()) {
    ui->spinTop->blockSignals(true);
    ui->spinTop->setValue(ui->spinBottom->value());
    ui->spinTop->blockSignals(false);
  }
  updateEditorFromSpinBoxes();
}

void CropInputTab::onBottomCropValueChanged() {
  if (ui->spinTop->value() > ui->spinBottom->value()) {
    ui->spinBottom->blockSignals(true);
    ui->spinBottom->setValue(ui->spinTop->value());
    ui->spinBottom->blockSignals(false);
  }
  updateEditorFromSpinBoxes();
}
