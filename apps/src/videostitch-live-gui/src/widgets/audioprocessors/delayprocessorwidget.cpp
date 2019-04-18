// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "delayprocessorwidget.hpp"
#include "ui_delayprocessorwidget.h"
#include "videostitcher/liveaudioprocessordelay.hpp"
#include "videostitcher/liveprojectdefinition.hpp"

DelayProcessorWidget::DelayProcessorWidget(LiveAudioProcessorDelay *ref, QWidget *parent)
    : AudioProcessorConfigurationWidget(parent), delayRef(ref), ui(new Ui::DelayProcessorWidget) {
  ui->setupUi(this);
  ui->verticalLayout->addLayout(buttonsLayout);
  connect(ui->sliderDelayValue, &QSlider::valueChanged, ui->spinDelay, &QSpinBox::setValue);
  connect(ui->spinDelay, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), ui->sliderDelayValue,
          &QSlider::setValue);
  connect(ui->spinDelay, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
          &DelayProcessorWidget::onSliderChanged);
  ui->spinDelay->setValue(delayRef->getDelay());
}

DelayProcessorWidget::~DelayProcessorWidget() { delete ui; }

LiveAudioProcessFactory *DelayProcessorWidget::getConfiguration() const { return delayRef; }

void DelayProcessorWidget::loadConfiguration() {
  if (projectDefinition) {
    ui->sliderDelayValue->setValue(projectDefinition->getAudioDelay());
  }
}

void DelayProcessorWidget::reactToChangedProject() { loadConfiguration(); }

void DelayProcessorWidget::onSliderChanged(const int value) {
  delayRef->setDelay(value);
  emit notifyConfigurationChanged(delayRef);
}
