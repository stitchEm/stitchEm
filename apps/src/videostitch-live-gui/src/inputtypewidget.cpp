// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "inputtypewidget.hpp"
#include "ui_inputtypewidget.h"

#include "guiconstants.hpp"

#include "libvideostitch-gui/mainwindow/vssettings.hpp"

#include <QDebug>

InputTypeWidget::InputTypeWidget(QWidget* parent) : QWidget(parent), ui(new Ui::InputTypeWidget) {
  ui->setupUi(this);
  ui->labelInputsTitle->setProperty("vs-title1", true);
  ui->inputTypeList->setProperty("vs-section-container", true);
  connect(ui->inputTypeList, &QListWidget::itemClicked, this, &InputTypeWidget::selectInputType);

  for (VideoStitch::InputFormat::InputFormatEnum inputType : getAvailableInputs()) {
    addInputType(inputType);
  }
}

InputTypeWidget::~InputTypeWidget() {}

void InputTypeWidget::addInputType(VideoStitch::InputFormat::InputFormatEnum inputType) {
  QString readableName = VideoStitch::InputFormat::getDisplayNameFromEnum(inputType);
  QListWidgetItem* item = new QListWidgetItem(readableName, ui->inputTypeList);
  item->setSizeHint(QSize(ui->inputTypeList->width(), ITEM_HEIGHT));
  item->setData(Qt::UserRole, int(inputType));
  ui->inputTypeList->addItem(item);
}

QVector<VideoStitch::InputFormat::InputFormatEnum> InputTypeWidget::getAvailableInputs() {
  // We are forcing the input plugins that we want to release
  QVector<VideoStitch::InputFormat::InputFormatEnum> availableInputs =
      QVector<VideoStitch::InputFormat::InputFormatEnum>()
      << VideoStitch::InputFormat::InputFormatEnum::PROCEDURAL <<
#if defined Q_OS_WIN
      VideoStitch::InputFormat::InputFormatEnum::MAGEWELL << VideoStitch::InputFormat::InputFormatEnum::MAGEWELLPRO
      << VideoStitch::InputFormat::InputFormatEnum::DECKLINK << VideoStitch::InputFormat::InputFormatEnum::AJA
      << VideoStitch::InputFormat::InputFormatEnum::XIMEA <<
#elif defined Q_OS_LINUX
      VideoStitch::InputFormat::InputFormatEnum::V4L2 <<
#endif
      VideoStitch::InputFormat::InputFormatEnum::NETWORK;
  if (VSSettings::getSettings()->getShowExperimentalFeatures()) {
    availableInputs << VideoStitch::InputFormat::InputFormatEnum::MEDIA;
  }
  return availableInputs;
}

void InputTypeWidget::selectInputType(QListWidgetItem* item) {
  emit inputTypeSelected(VideoStitch::InputFormat::InputFormatEnum(item->data(Qt::UserRole).toInt()));
}
