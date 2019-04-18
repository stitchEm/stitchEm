// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "multiinputconfiguration.hpp"
#include "ui_multiinputconfiguration.h"
#include "newinputitemwidget.hpp"

#include "videostitcher/liveprojectdefinition.hpp"

#include <QPushButton>

MultiInputConfiguration::MultiInputConfiguration(std::shared_ptr<const LiveInputStream> liveInput,
                                                 QWidget* const parent)
    : InputConfigurationWidget(parent),
      ui(new Ui::MultiInputConfiguration),
      inputsChecked(0),
      templateInput(liveInput) {
  ui->setupUi(this);
  ui->labelTitle->setProperty("vs-title1", true);
  ui->verticalLayout->addLayout(buttonsLayout);
  ui->frameError->hide();
  connect(ui->inputNbSpinBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
          &MultiInputConfiguration::onConfigurationChanged);
}

MultiInputConfiguration::~MultiInputConfiguration() {}

void MultiInputConfiguration::onUrlsValidityChanged() {
  ++inputsChecked;
  if (inputsChecked == ui->listInputs->count()) {
    ui->inputNbSpinBox->setEnabled(true);
    ui->linePattern->setEnabled(true);
    const bool valid = streamsAreValid();
    if (valid) {
      IConfigurationCategory::save();
    }
    ui->frameError->setVisible(!valid);
    inputsChecked = 0;
  }
}

void MultiInputConfiguration::onTestAllInputs() {
  ui->inputNbSpinBox->setEnabled(false);
  ui->linePattern->setEnabled(false);
  for (NewInputItemWidget* widget : getItemWidgets()) {
    widget->onTestClicked();
  }
}

void MultiInputConfiguration::changeNbOfInputs(int newNbOfInputs) {
  const auto oldNbOfInputs = ui->listInputs->count();
  if (newNbOfInputs < oldNbOfInputs) {
    for (auto index = oldNbOfInputs - 1; index >= newNbOfInputs; --index) {
      delete ui->listInputs->itemWidget(ui->listInputs->item(index));
      delete ui->listInputs->takeItem(index);
    }
  } else {
    for (auto index = oldNbOfInputs; index < newNbOfInputs; ++index) {
      addInputWidget(ui->linePattern->text(), index);
    }
  }
  adjustListSize();
}

bool MultiInputConfiguration::streamsAreValid() {
  bool valid = true;
  for (auto widget : getItemWidgets()) {
    valid = valid && widget->hasValidUrl();
  }
  return valid;
}

void MultiInputConfiguration::adjustListSize() {
  const auto maxHeight = height() / 2;
  const auto height = ui->listInputs->count() * 40;
  ui->listInputs->setFixedHeight(qMin(maxHeight, height));
}

QVector<NewInputItemWidget*> MultiInputConfiguration::getItemWidgets() const {
  QVector<NewInputItemWidget*> widgets;
  for (auto index = 0; index < ui->listInputs->count(); ++index) {
    auto item = ui->listInputs->item(index);
    widgets.push_back(qobject_cast<NewInputItemWidget*>(ui->listInputs->itemWidget(item)));
  }
  return widgets;
}

void MultiInputConfiguration::saveData() {
  editedInputs.clear();
  for (auto widget : getItemWidgets()) {
    std::shared_ptr<LiveInputStream> input(new LiveInputStream(widget->getUrl()));
    input->setWidth(widget->property("frame-width").toLongLong());
    input->setHeight(widget->property("frame-height").toLongLong());
    editedInputs.append(input);
  }
}

void MultiInputConfiguration::reactToChangedProject() {
  disconnect(ui->inputNbSpinBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
             &MultiInputConfiguration::changeNbOfInputs);
  QStringList inputNames = projectDefinition->getVideoInputNames();
  if (inputNames.isEmpty()) {
    // At least 1 input for the creation
    inputNames.append(ui->linePattern->text());
  }

  const auto nbInputs = inputNames.count();
  ui->inputNbSpinBox->setValue(nbInputs);
  for (auto index = 0; index < nbInputs; ++index) {
    addInputWidget(inputNames.at(index), index);
  }
  adjustListSize();
  connect(ui->inputNbSpinBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
          &MultiInputConfiguration::changeNbOfInputs);
}

bool MultiInputConfiguration::save() {
  ui->frameError->hide();
  onTestAllInputs();
  return false;
}

void MultiInputConfiguration::addInputWidget(QString inputName, int index) {
  NewInputItemWidget* widget = new NewInputItemWidget(inputName, index, this);
  QListWidgetItem* item = new QListWidgetItem(ui->listInputs);
  item->setSizeHint(QSize(ui->listInputs->width(), widget->height()));
  ui->listInputs->addItem(item);
  ui->listInputs->setItemWidget(item, widget);
  connect(ui->linePattern, &QLineEdit::textEdited, widget, &NewInputItemWidget::setUrl);
  connect(widget, &NewInputItemWidget::notifyUrlValidityChanged, this, &MultiInputConfiguration::onUrlsValidityChanged);
  connect(widget, &NewInputItemWidget::notifyUrlContentChanged, this, &MultiInputConfiguration::onConfigurationChanged);
  connect(widget, &NewInputItemWidget::notifyTestActivated, this, &MultiInputConfiguration::notifyTestActivated);
}

void MultiInputConfiguration::onInputTestResult(const int id, const bool result, qint64 width, qint64 height) {
  for (auto widget : getItemWidgets()) {
    if (widget->getId() == id) {
      // We must keep this size, it's the real size of the stream
      widget->setProperty("frame-width", width);
      widget->setProperty("frame-height", height);
      QMetaObject::invokeMethod(widget, "onTestFinished", Qt::DirectConnection, Q_ARG(const bool, result));
      break;
    }
  }
}
