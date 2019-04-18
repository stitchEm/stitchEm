// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "outputconfigurationwidget.hpp"

#include "videostitcher/liveoutputfactory.hpp"

OutputConfigurationWidget::OutputConfigurationWidget(QWidget* parent) : IConfigurationCategory(parent) {}

void OutputConfigurationWidget::fillWidgetWithValue() {
  if (getOutput()->getOutputState() != LiveOutputFactory::OutputState::DISABLED) {
    toggleWidgetState();
  }
}
